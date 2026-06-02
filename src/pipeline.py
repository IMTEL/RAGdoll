import json
import re
import time
import uuid

from src.llm import create_llm
from src.models import Agent
from src.models.chat.command import Command
from src.rag_service.dao import get_context_dao
from src.rag_service.embeddings import (
    GoogleEmbedding,
    OpenAIEmbedding,
)


CONTEXT_TRUNCATE_LENGTH = 500  # Limit context text length to avoid overly long prompts


def _normalize_function_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "", name.strip())


def _extract_json_object(text: str) -> dict | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _parse_llm_response(raw_response: str, allowed_function_names: set[str]) -> tuple[str, list[dict]]:
    parsed = _extract_json_object(raw_response)
    if parsed is not None:
        message = parsed.get("message")
        calls = parsed.get("functions", [])
        function_calls = []
        if isinstance(calls, list):
            for call in calls:
                if not isinstance(call, dict):
                    continue
                name = _normalize_function_name(str(call.get("name", "")))
                if not name or name not in allowed_function_names:
                    continue
                arguments = call.get("arguments", {})
                if not isinstance(arguments, dict):
                    arguments = {}
                function_calls.append({"name": name, "arguments": arguments})
        return str(message or ""), function_calls

    function_calls = []
    parsed_response = raw_response
    for function_match in re.finditer(
        r"\[FUNCTION\](.*?)\|(.*?)\[\/FUNCTION\]", raw_response, re.DOTALL
    ):
        function_name = _normalize_function_name(function_match.group(1))
        if function_name and function_name in allowed_function_names:
            function_calls.append(
                {
                    "name": function_name,
                    "arguments": {"value": function_match.group(2).strip()},
                }
            )
        parsed_response = parsed_response.replace(function_match.group(0), "").strip()

    return parsed_response.strip(), function_calls


def function_prompt_section(agent: Agent, active_role_id: str | None) -> tuple[str, set[str]]:
    if not active_role_id:
        return "", set()

    role = agent.get_role_by_name(active_role_id)
    if role is None or not role.function_access:
        return "", set()

    function_definitions = [
        function
        for function in agent.functions
        if function.name in role.function_access
    ]
    if not function_definitions:
        return "", set()

    allowed_names = {_normalize_function_name(function.name) for function in function_definitions}
    lines = [
        "You may request external function calls, but only from this allowed list.",
        "Always respond as valid JSON and nothing else, using this exact shape:",
        '{"message":"visible response to the user","functions":[{"name":"function_name","arguments":{"field":"value"}}]}',
        "If no function should be called, return an empty functions array.",
        "The message should be conversational and should not mention implementation details unless useful.",
        "Only call a function when its call instructions are satisfied.",
        "Available functions:",
    ]

    for function in function_definitions:
        lines.append(f"- name: {function.name}")
        lines.append(f"  required_fields: {function.required_fields}")
        lines.append(f"  call_instructions: {function.call_instructions}")
        if function.explanation:
            lines.append(f"  explanation: {function.explanation}")
        if function.example_output:
            lines.append(f"  example_output: {function.example_output}")

    return "\n".join(lines), allowed_names


def generate_retrieval_query(
    command: Command, agent: Agent, role_prompt: str = ""
) -> str:
    """Generate a standalone query for RAG retrieval from conversation context.

    This function uses an LLM to synthesize the latest user message along with
    recent conversation context into a single, focused query that can be used
    for semantic search.

    Args:
        command: Command object containing chat history and context
        agent: Agent object containing LLM provider configuration
        role_prompt: Optional role description for context

    Returns:
        A standalone query string optimized for RAG retrieval
    """
    if not command.chat_log:
        return ""
    last_message = command.chat_log[-1].content if command.chat_log else ""
    recent_context = chat_history_prompt_section(
        command, limit=6, include_header=False, include_latest=True
    )

    if not recent_context:
        recent_context = "No prior context"

    summary_prompt = (
        f"Rewrite this user's latest message as a standalone message that captures all necessary context. This will be used to retrieve relevant information from a knowledge base."
        'Replace any pronouns ("you", "your", "it", "that", etc.) with what the message is referencing from the context.'
        'Do not reference the context, do not use pronouns ("you", "your", "it", "that", etc.), do not reference the role prompt or character prompt, ALWAYS include the information directly in the rewritten message.'
        f"User message to rewrite: {last_message}\n\n"
        "CONTEXT:\n"
        f"Prompt given to the character the user is talking to: {agent.prompt}\n"
        f"Their role: {role_prompt if role_prompt else 'unspecified'}\n"
        f"Recent conversation context:\n{recent_context}\n\n"
        f"Standalone message:"
    )

    try:
        query_llm = create_llm(agent.llm_provider, agent.llm_model, agent.llm_api_key)
        standalone_query = query_llm.generate(
            summary_prompt, agent.llm_max_tokens, agent.llm_temperature
        )
        print(f"Generated retrieval query: {standalone_query}")
        return standalone_query.strip()
    except Exception as e:
        print(f"Error generating retrieval query: {e}")
        return last_message


def assemble_prompt_with_agent(command: Command, agent: Agent) -> dict:
    """Assemble a prompt using agent configuration and role-based RAG.

    This function:
    1. Uses the agent's system prompt as the base
    2. Retrieves context from the corpus accessible by the active roles
    3. Generates a response using the agent's configured LLM

    Args:
        command: The user command with conversation history and context
        agent: The agent configuration to use

    Returns:
        Dictionary with response, function calls, and metadata
    """
    # Use agent's prompt as base, with variable substitution
    full_chat_history = chat_history_prompt_section(command, include_latest=True)
    full_chat_history = chat_history_prompt_section(command)

    # Assemble final prompt
    role_prompt = (
        agent.get_role_by_name(command.active_role_id).description
        if command.active_role_id
        else None
    )
    function_prompt, allowed_function_names = function_prompt_section(
        agent, command.active_role_id
    )
    last_user_response = command.chat_log[-1].content if command.chat_log else ""

    # Determine retrieval query based on agent.context_aware_retrieval
    if hasattr(agent, "context_aware_retrieval") and not agent.context_aware_retrieval:
        # Only use the latest chat message as the retrieval query
        retrieval_query = command.chat_log[-1].content if command.chat_log else ""
    else:
        # Use context-aware retrieval (default)
        retrieval_query = generate_retrieval_query(
            command=command, agent=agent, role_prompt=role_prompt if role_prompt else ""
        )

    # Get accessible categories based on active roles
    accessible_documents = (
        agent.get_role_by_name(command.active_role_id).document_access
        if command.active_role_id
        else []
    )

    print(f"Active role: {command.active_role_id}")
    print(f"Accessible documents for role: {accessible_documents}")
    print(f"Number of accessible documents: {len(accessible_documents)}")

    # Perform RAG retrieval from accessible documents
    retrieved_contexts = []
    db = get_context_dao()

    # Use agent's configured embedding model
    embedding_model_config = agent.embedding_model
    print("Using embedding model:", embedding_model_config)

    # Parse provider:model format (e.g., "openai:text-embedding-3-small" or "gemini:text-embedding-004")
    if ":" in embedding_model_config:
        provider, model_name = embedding_model_config.split(":", 1)
    else:
        # Fallback: if no colon use default model
        provider = "gemini"
        model_name = "text-embedding-004"

    # Create the appropriate embedding model based on provider
    embedding_api_key = getattr(agent, "embedding_api_key", None)
    if provider.lower() == "openai":
        embedding_model = OpenAIEmbedding(
            model_name=model_name, embedding_api_key=embedding_api_key
        )
    elif provider.lower() in ["google", "gemini"]:
        embedding_model = GoogleEmbedding(
            model_name=model_name, embedding_api_key=embedding_api_key
        )
    else:
        # Fallback to Gemini if provider not recognized
        embedding_model = GoogleEmbedding(
            model_name="text-embedding-004", embedding_api_key=embedding_api_key
        )

    try:
        # Generate embedding for the standalone retrieval query
        embeddings: list[float] = embedding_model.get_embedding(retrieval_query)

        # Retrieve relevant contexts for the agent with optional category filtering
        # Use the standalone query for both semantic and keyword search
        retrieved_contexts = db.get_context_for_agent(
            agent_id=agent.id
            if agent.id
            else "",  # TODO: Raise error if agent.id is None
            query_embedding=embeddings,
            query_text=retrieval_query,  # Standalone query for semantic search
            keyword_query_text=retrieval_query,  # Standalone query for BM25
            documents=accessible_documents,
            top_k=agent.top_k,  # Use agent's configured top_k
            similarity_threshold=agent.similarity_threshold,  # Use agent's configured threshold
            hybrid_search_alpha=agent.hybrid_search_alpha,  # Use agent's configured alpha
        )
    except Exception as e:
        print(f"Error retrieving context: {e}")
        retrieved_contexts = []

    print(f"Retrieved {len(retrieved_contexts)} contexts for agent {agent.name}")

    prompt = ""
    if last_user_response:
        prompt += "\nRESPOND TO THIS NEW USER MESSAGE: " + last_user_response + "\n"
    prompt += (
        "You are playing a character, and should respond as that character. This is your prompt: "
        + agent.prompt
        + "\n"
    )
    prompt += (
        "Return only the character's spoken response. Do not prefix the answer "
        "with AGENT:, ASSISTANT:, the character name, or any role label.\n"
    )
    prompt += ("Your role: " + role_prompt + "\n") if role_prompt else ""
    if function_prompt:
        prompt += "\n\nExternal function calling instructions:\n" + function_prompt + "\n"
    # Add retrieved context to prompt
    if retrieved_contexts:
        prompt += "\n\nRelevant Information (IMPORTANT: this information is 100% true for your role in your world, prioritise it over all other sources):\n"
        # for idx, ctx in enumerate(retrieved_contexts, 1):
        #     prompt += f"\n[Context {idx} from {ctx.document_name}]:\n{ctx.text}\n"
        for ctx in retrieved_contexts:
            prompt += f"\n-{ctx.text}\n"

    game_context = game_context_prompt_section(command)
    if game_context:
        prompt += "\n\n" + game_context

    prompt = full_chat_history + "\n" + prompt

    print(f"Prompt sent to LLM:\n{prompt}")

    # Use agent's configured LLM with specific model
    language_model = create_llm(
        llm_provider=agent.llm_provider,
        model=agent.llm_model,
        api_key=agent.llm_api_key,
    )
    response = language_model.generate(prompt)

    parsed_response, function_calls = _parse_llm_response(
        response, allowed_function_names
    )
    legacy_function_call = (
        {
            "function_name": function_calls[0]["name"],
            "function_parameters": [function_calls[0]["arguments"]],
        }
        if function_calls
        else None
    )

    return {
        "id": str(uuid.uuid4()),
        "created": int(time.time()),
        "model": agent.llm_model,
        "agent_id": command.agent_id,
        "active_role": command.active_role_id,
        "accessible_documents": accessible_documents,
        "context_used": [
            {
                "document_name": ctx.document_name,
                "chunk_index": ctx.chunk_index,
                "content": ctx.text[:CONTEXT_TRUNCATE_LENGTH],
            }
            for ctx in retrieved_contexts
        ],
        "metadata": {
            "response_length": len(parsed_response),
            "agent_name": agent.name,
            "num_context_retrieved": len(retrieved_contexts),
            "num_progress_items": len(command.progress),
        },
        "function_call": legacy_function_call,
        "function_calls": function_calls,
        "response": parsed_response,
    }


def chat_history_prompt_section(
    command, limit: int = 100, include_header: bool = True, include_latest: bool = False
) -> str:
    chat_history = ""
    if len(command.chat_log) > 1:
        if include_header:
            chat_history += "This is the previous conversation:\n"
        limit_start = max(0, len(command.chat_log) - limit - 1)
        for msg in command.chat_log[
            limit_start : (-1 if not include_latest else None)
        ]:  # Exclude latest user message
            role_label = "ASSISTANT" if msg.role.lower() == "agent" else msg.role.upper()
            chat_history += f"{role_label}: {msg.content}\n"
    return chat_history


def game_context_prompt_section(command: Command) -> str:
    sections: list[str] = []

    if command.user_information:
        sections.append(
            "User/game information:\n"
            + "\n".join(f"- {item}" for item in command.user_information if item)
        )

    if command.user_actions:
        sections.append(
            "Recent user/game actions:\n"
            + "\n".join(f"- {action}" for action in command.user_actions if action)
        )

    if command.progress:
        progress_lines: list[str] = []
        for task in command.progress:
            progress_lines.append(
                f"- {task.task_name}: {task.status}. {task.description}"
            )
            for subtask in task.subtask_progress:
                state = "complete" if subtask.completed else "incomplete"
                progress_lines.append(
                    f"  - {subtask.subtask_name}: {state}. {subtask.description}"
                )
                for step in subtask.step_progress:
                    step_state = "complete" if step.completed else "incomplete"
                    progress_lines.append(
                        f"    - {step.step_name} repetition {step.repetition_number}: {step_state}"
                    )
        if progress_lines:
            sections.append("Training/task progress:\n" + "\n".join(progress_lines))

    if not sections:
        return ""

    return (
        "Additional live context from the external application. Use it only when it is relevant "
        "to the user's question, the user's current task, or the character's role. "
        "Do not mention task progress unless it helps answer the user or guide their next action. Tasks may be incomplete, do not assume that it is done entirely unless status is finished, as the user may need guidance in finishing the task:  :\n"
        + "\n\n".join(sections)
        
    )
