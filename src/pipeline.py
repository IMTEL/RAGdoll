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
        query_llm = create_llm(agent.llm_provider, agent.llm_model)
        standalone_query = query_llm.generate(summary_prompt,agent.llm_max_tokens,agent.llm_temperature)
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
    last_user_response = command.chat_log[-1].content if command.chat_log else ""

    # Generate a standalone query for RAG retrieval using LLM
    retrieval_query = generate_retrieval_query(
        command=command, agent=agent, role_prompt=role_prompt if role_prompt else ""
    )

    # Get accessible categories based on active roles
    accessible_documents = (
        agent.get_role_by_name(command.active_role_id).document_access
        if command.active_role_id
        else []
    )

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
    if provider.lower() == "openai":
        embedding_model = OpenAIEmbedding(model_name=model_name)
    elif provider.lower() in ["google", "gemini"]:
        embedding_model = GoogleEmbedding(model_name=model_name)
    else:
        # Fallback to Gemini if provider not recognized
        embedding_model = GoogleEmbedding(model_name="text-embedding-004")

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
    prompt += ("Your role: " + role_prompt + "\n") if role_prompt else ""
    # Add retrieved context to prompt
    if retrieved_contexts:
        prompt += "\n\nRelevant Information (IMPORTANT: this information is 100% true for your role in your world, prioritise it over all other sources):\n"
        # for idx, ctx in enumerate(retrieved_contexts, 1):
        #     prompt += f"\n[Context {idx} from {ctx.document_name}]:\n{ctx.text}\n"
        for ctx in retrieved_contexts:
            prompt += f"\n-{ctx.text}\n"

    prompt = full_chat_history + "\n" + prompt

    print(f"Prompt sent to LLM:\n{prompt}")

    # Use agent's configured LLM with specific model
    language_model = create_llm(llm_provider=agent.llm_provider, model=agent.llm_model)
    response = language_model.generate(prompt)

    # Parse function calls from response
    function_call = None
    parsed_response = response

    function_match = re.search(
        r"\[FUNCTION\](.*?)\|(.*?)\[\/FUNCTION\](.*)", response, re.DOTALL
    )
    if function_match:
        function_name = function_match.group(1).strip()
        function_param = function_match.group(2).strip()
        function_tag_text = function_match.group(0)
        parsed_response = response.replace(function_tag_text, "").strip()

        function_call = {
            "function_name": function_name,
            "function_parameters": [function_param],
        }

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
        },
        "function_call": function_call,
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
            chat_history += f"{msg.role.upper()}: {msg.content}\n"
    return chat_history
