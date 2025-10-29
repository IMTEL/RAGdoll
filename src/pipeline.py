import logging
import re
import time
import uuid

from src.config import Config
from src.llm import create_llm
from src.models import Agent
from src.models.chat.command import Command
from src.rag_service.dao import get_context_dao
from src.rag_service.embeddings import GoogleEmbedding, create_embeddings_model


logger = logging.getLogger(__name__)


CONTEXT_TRUNCATE_LENGTH = 500  # Limit context text length to avoid overly long prompts


def get_answer_from_user(
    answer: str, target: str, question: str, model="gemini"
) -> str:
    """Get the answer from the user. Target is what the question is about. Example: "What is your name?" -> target= "name"."""
    prompt = ""
    if target == "name":
        prompt = """A user has provided the following answer to the question: {question}. 
                        The answer is: {answer}. The question is about the user's {target} You are to ONLY REPLY IN JSON FORMAT like so:
                        target: "some answer"
                        An example of a valid response is for the question "What is your name?" where target is name and answer from user is My name is John Doe is:
                        name: "John Doe"
                    """
    else:
        prompt = """A user has provided the following answer to the question: {question}. 
                        The answer is: {answer}. The question is about the user's {target} You are to ONLY REPLY IN JSON FORMAT like so:
                        target: "some answer"
                        When the target is user_mode, you shall review the answer and categorize it as either "experienced, beginner, or unsure".
                        An example of a valid response is for the question "What is your name?" where target is user_mode and answer from user is "I am not experienced with VR" is:
                        user_mode: "beginner"
                    """
    prompt.format(answer=answer, target=target, question=question)
    # skoleelev
    # unge arbeidsledige
    # studerer fagskole
    # om de har jobberfaring
    # Nettopp ferdig med skole?
    # arbeidsløs en stund
    # har de vært i jobb før?
    # har de hatt praksis tidligere i et relevant felt?
    # interests?

    language_model = create_llm(model)
    response = language_model.generate(prompt)
    if response is None:
        return "No response from the language model."
    if response == "":
        return "Empty response from the language model."

    return response


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
    embed_chat_history = chat_history_prompt_section(command, limit = 3, include_header = False, include_latest = True)
    full_chat_history = (
        chat_history_prompt_section(command)
    )

    # Assemble final prompt
    role_prompt = "Role: " + (
        agent.get_role_by_name(command.active_role_id).description
        if command.active_role_id
        else None
    )
    last_user_response = command.chat_log[-1].content if command.chat_log else ""

    to_embed: str = (
        f"CURRENT USER QUESTION: {last_user_response}\n\n"
        f"CURRENT USER QUESTION (emphasis): {last_user_response}\n\n"
        f"Agent info: {agent.prompt}\n\n"
        f"Role info: {role_prompt if role_prompt else 'none'}\n\n"
        f"Conversation Context:\n{embed_chat_history}"
    )

    print(f"Embedding text for retrieval:\n{to_embed}")

    # Get accessible categories based on active roles
    accessible_documents = agent.get_role_by_name(command.active_role_id).document_access if command.active_role_id else []

    print("Accessible documents for role:", accessible_documents, command.active_role_id)

    # Perform RAG retrieval from accessible documents
    retrieved_contexts = []
    db = get_context_dao()

    # Use agent's configured embedding model, fallback to "google"
    # embedding_model_name = getattr(agent, "embedding_model", "google")
    # print("Using embedding model:", embedding_model_name)
    # embedding_model = create_embeddings_model(embedding_model_name)
    # TODO: Change to use agent's configured embedding model when we have more than one
    embedding_model = GoogleEmbedding()

    try:
        # Generate embedding for the user's query
        embeddings: list[float] = embedding_model.get_embedding(to_embed)

        # Retrieve relevant contexts for the agent with optional category filtering
        retrieved_contexts = db.get_context_for_agent(
            agent_id=agent.id
            if agent.id
            else "",  # TODO: Raise error if agent.id is None
            embedding=embeddings,
            documents=accessible_documents,
            top_k=3,  # Retrieve top 3 most relevant contexts
        )
    except Exception as e:
        print(f"Error retrieving context: {e}")
        retrieved_contexts = []

    print(f"Retrieved {len(retrieved_contexts)} contexts for agent {agent.name}")

    prompt = "INSTRUCTIONS: " + agent.prompt
    prompt += ("\n" + role_prompt) if role_prompt else ""
    # Add retrieved context to prompt
    if retrieved_contexts:
        prompt += "\n\nRelevant Information (IMPORTANT: this information is 100% true for your role in your universe, prioritise it over all other sources):\n"
        # for idx, ctx in enumerate(retrieved_contexts, 1):
        #     prompt += f"\n[Context {idx} from {ctx.document_name}]:\n{ctx.text}\n"
        for ctx in retrieved_contexts:
            prompt += f"\n-{ctx.text}\n"

    if last_user_response:
        prompt += "\nRESPOND TO THIS NEW USER MESSAGE: " + last_user_response

    prompt = full_chat_history + "\n" + prompt

    print(f"Prompt sent to LLM:\n{prompt}")

    # Define llm_provider from agent's configuration
    llm_provider = agent.llm_provider
    logger.info(f"Using LLM provider: {llm_provider}")

    # Use agent's configured LLM
    language_model = create_llm(llm_provider)
    
    try:
        logger.info(f"Sending prompt to {llm_provider} (length: {len(prompt)} chars)")
        response = language_model.generate(prompt)
        if not response:
            raise ValueError("Empty response from LLM")
        logger.info(f"Received response from {llm_provider} (length: {len(response)} chars)")
    except Exception as e:
        logger.error(f"Error generating response from LLM ({llm_provider}): {e}")
        response = "I apologize, but I'm having trouble generating a response right now. Please try again in a moment."

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
        "model": llm_provider,
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

def chat_history_prompt_section(command, limit: int = 100, include_header: bool = True, include_latest: bool = False) -> str:
    chat_history = ""
    if len(command.chat_log) > 1:
        if include_header:
            chat_history += "This is the previous conversation:\n"
        limit_start = max(0, len(command.chat_log) - limit - 1)
        for msg in command.chat_log[limit_start:(-1 if not include_latest else None)]:  # Exclude latest user message
            chat_history += f"{msg.role.upper()}: {msg.content}\n"
    return chat_history


def assemble_prompt(command: Command, model: str = Config().MODEL) -> dict:
    """Assembles a prompt for a large language model and prompt LLM to generate a response."""
    # to_embed: str = str(command.question) + " "+ str(command.progress) + " "+ str(command.user_actions)

    # to_embed: str = str(command.chat_log[-1].content)
    to_embed: str = (
        str(command.chat_log[-1].content) if command.chat_log else "No user message"
    )

    db = get_context_dao()
    embedding_model = create_embeddings_model()
    embeddings: list[float] = embedding_model.get_embedding(to_embed)
    context = db.get_context(
        "hello", embeddings
    )  # TODO: remove document name from here

    base_prompt = """
    MANDATORY: 
    If you detect the user needs an action, you MUST guess the appropriate function call using [FUNCTION]function_name|parameter[/FUNCTION].
    If you are unsure, make your best guess. DO NOT return an empty response.

    You are a helpful assistant and guide in the Blue Sector Virtual Reality work training. 
    You are here to help the user with their questions and guide them through the training.

    Earlier chathistory is: {command.chat_log}
    The user is currently in the {command.scene_name} scene. When the user asks a question from {command.scene_name} = ReceptionOutdoor, your name is Rachel. When the user asks a question from {command.scene_name} = Laboratory, your name is Larry.

    The information you have obtained on the user is {command.user_information}. ADJUST YOUR ANSWER BASED ON THIS, IF IT IS AVAILABLE. IF TWO ANSWERS TO THE SAME QUESTION ARE GIVEN, USE THE LATEST ONE.
    IF TWO ANSWERS CONTRADICT EACH OTHER, USE THE LATEST ONE.
    If user information is unavailable, try to provide a general answer.
    The user has made the following progress: {command.progress}.
    The user has taken the following actions: {command.user_actions}. (Actions may not be available) Refer to these when a user asks about an object they have interacted with.
    IF THERE ARE NO CONTEXT AVAILABLE, PLEASE STATE THAT YOU ARE NOT SURE, BUT TRY TO PROVIDE AN ANSWER.
    PROVIDE A SHORT ANSWER THAT IS EASY TO UNDERSTAND. STATE THE NAME OF THE USER IN A NATURAL WAY IN THE RESPONSE.

    IMPORTANT - FUNCTION CALLING INSTRUCTIONS:
    When the user asks to move to a different location or scene, or when you need to assist them by showing objects or playing animations,
    follow this two-step confirmation process:

    1. First, ask the user to confirm the action you're about to perform. For example:
       "Would you like me to teleport you to the Laboratory?" or "Shall I show you the safety equipment?"
    
    2. Only after the user confirms, use the function calling format as your next response:
       [FUNCTION]function_name|parameter[/FUNCTION]
    
    If the user has already explicitly confirmed in their current message, you may execute the function immediately.

    IMPORTANT CONSTRAINTS:
    - NEVER teleport the user to the scene they are currently in. If they ask to go to their current location, inform them they are already there.

    Function call format:
    [FUNCTION]function_name|parameter[/FUNCTION]

    Available functions:
    - teleport(location: str): Teleport the user to a specific scene. Available scenes are:
        * ReceptionOutdoor - The main reception area
        * Laboratory - Where scientific experiments take place
    If the user asks to go to any of these locations or expresses a desire to move to another area, follow the confirmation process.
    REMEMBER: Do not teleport the user to their current scene location.

    - showObject(objectId: str): Highlight or show a specific object to the user.

    Example 1: If the user says "Can you take me to the lab?", respond with:
    "Would you like me to teleport you to the Laboratory? Please confirm."
    
    Example 2: If the user then confirms "Yes, please", respond with:
    "[FUNCTION]teleport|Laboratory[/FUNCTION]"
    
    Example 3: If the user says "Yes, teleport me to the reception right now", you can respond directly with:
    "[FUNCTION]teleport|ReceptionOutdoor[/FUNCTION]"

    Example 4: If the user says "Take me to the Laboratory" but they are already in the Laboratory scene, respond with:
    "You're already in the Laboratory. Is there something specific you're looking for here?"
    
    Answer in a SHORT and UNDERSTANDABLE way, NOT exceeding 200 characters.
    """

    base_prompt = base_prompt.format(command=command)
    prompt: str = ""
    if context is None or len(context) == 0:
        prompt += (
            str(base_prompt)
            + "context: NO CONTEXT AVAILABLE "
            + "question: "
            + str(command.chat_log[-1])
        )

    else:
        prompt += (
            str(base_prompt)
            + "context: "
            + str(context[0])
            + "question: "
            + str(command.chat_log[-1])
        )

    print(f"Prompt sent to LLM:\n{prompt}")

    language_model = create_llm(model)
    response = language_model.generate(prompt)

    # Parse the response for function calls
    function_call = None
    parsed_response = response

    import re

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
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": parsed_response,
                    "function_call": function_call,
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            # "prompt_tokens": prompt_token_count,
            # "completion_tokens": completion_token_count,
            # "total_tokens": total_tokens
        },
        "system_fingerprint": "v1-system",  # placeholder
        "context_used": context if context else [],
        "metadata": {
            "response_length": len(parsed_response),
            # "confidence_score": 0.95
        },
        "response": parsed_response,
        "function_call": function_call,
    }
