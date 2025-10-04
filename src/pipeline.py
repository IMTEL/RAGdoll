import re
import time
import uuid

from src.command import Command
from src.config import Config
from src.domain.agents import Agent
from src.llm import create_llm
from src.rag_service.embeddings import create_embeddings_model
from src.rag_service.repositories import get_context_repository


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


def assemble_prompt_with_agent(
    command: Command, agent: Agent, model: str | None = None
) -> dict:
    """Assemble a prompt using agent configuration and role-based RAG.

    This function:
    1. Uses the agent's system prompt as the base
    2. Retrieves context from the corpus accessible by the active roles
    3. Generates a response using the agent's configured LLM

    Args:
        command: The user command with conversation history and context
        agent: The agent configuration to use
        model: Override LLM model (uses agent.llm_model if None)

    Returns:
        Dictionary with response, function calls, and metadata
    """
    # Extract the user's question from chat log
    to_embed: str = (
        str(command.chat_log[-1].content) if command.chat_log else "No user message"
    )

    # Get accessible corpus based on active roles
    accessible_corpus = agent.get_corpus_for_roles(command.active_role_ids)

    # If no roles specified, use all corpus
    if not command.active_role_ids and agent.corpa:
        accessible_corpus = agent.corpa

    # Perform RAG retrieval from accessible corpus
    db = get_context_repository()
    embedding_model = create_embeddings_model()
    embeddings: list[float] = embedding_model.get_embedding(to_embed)

    # TODO: Update context retrieval to filter by accessible_corpus
    # For now, retrieve context normally
    context = db.get_context("hello", embeddings)

    # Use agent's prompt as base, with variable substitution
    base_prompt = agent.prompt.format(
        chat_log=command.chat_log,
    )

    # Assemble final prompt
    if context is None or len(context) == 0:
        prompt = (
            str(base_prompt)
            + "\n\ncontext: NO CONTEXT AVAILABLE\n"
            + "question: "
            + str(command.chat_log[-1].content if command.chat_log else "")
        )
    else:
        prompt = (
            str(base_prompt)
            + "\n\ncontext: "
            + str(context[0])
            + "\nquestion: "
            + str(command.chat_log[-1].content if command.chat_log else "")
        )

    print(f"Prompt sent to LLM:\n{prompt}")

    # Use agent's configured LLM
    model_to_use = model or agent.llm_model
    language_model = create_llm(model_to_use)
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
        "model": model_to_use,
        "agent_id": command.agent_id,
        "active_roles": command.active_role_ids,
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
        "usage": {},
        "system_fingerprint": "v1-system",
        "context_used": context if context else [],
        "accessible_corpus": accessible_corpus,
        "metadata": {
            "response_length": len(parsed_response),
            "agent_name": agent.name,
        },
        "response": parsed_response,
        "function_call": function_call,
    }


def assemble_prompt(command: Command, model: str = Config().MODEL) -> dict:
    """Assembles a prompt for a large language model and prompt LLM to generate a response."""
    # to_embed: str = str(command.question) + " "+ str(command.progress) + " "+ str(command.user_actions)

    # to_embed: str = str(command.chat_log[-1].content)
    to_embed: str = (
        str(command.chat_log[-1].content) if command.chat_log else "No user message"
    )

    db = get_context_repository()
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
