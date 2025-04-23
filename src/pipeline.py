
from src.command import Command, Prompt, prompt_to_json
from src.rag_service.dao import get_database
from src.config import Config
from src.rag_service.embeddings import create_embeddings_model
from src.LLM import LLM, create_llm
import uuid
import time

def getAnswerFromUser(answer: str, target: str, question: str, model = "gemini") -> str:
    """Get the answer from the user. Target is what the question is about. Example: "What is your name?" -> target= "name"."""
    prompt = ""
    if target == "name":
    
        prompt = """A user has provided the following answer to the question: {question}. 
                        The answer is: {answer}. The question is about the user's {target} You are to ONLY REPLY IN JSON FORMAT like so:
                        target: "some answer"
                        An example of a valid response is for the question "What is your name?" where target is name and answer from user is My name is John Doe is:
                        name: "John Doe"
                    """
    else :
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

def assemble_prompt(command: Command, model: str = "openai") -> dict[str]:
    """Assembles a prompt for a large language model and prompt LLM to generate a response."""
    
    #to_embed: str = str(command.question) + " "+ str(command.progress) + " "+ str(command.user_actions)
    
    to_embed: str = str(command.chatLog[-1].content)
    
    db = get_database()
    embedding_model = create_embeddings_model()
    embeddings: list[float] = embedding_model.get_embedding(to_embed)
    context = db.get_context("hello", embeddings ) # TODO: remove document name from here
    
    base_prompt = """
    You are a helpful assistant and guide in the Blue Sector Virtual Reality work training. 
    You are here to help the user with their questions and guide them through the training.
    Earlier chathistory is: {command.chatLog}
    The information you have obtained on the user is {command.user_information}. ADJUST YOUR ANSWER BASED ON THIS, IF IT IS AVAILABLE.
    If user information is unavailable, try to provide a general answer.
    The user has made the following progress: {command.progress}.
    The user has taken the following actions: {command.user_actions}. (Actions may not be available)
    IF THERE ARE NO CONTEXT AVAILABLE, PLEASE STATE THAT YOU ARE NOT SURE, BUT TRY TO PROVIDE AN ANSWER.
    PROVIDE A SHORT ANSWER THAT IS EASY TO UNDERSTAND. STATE THE NAME OF THE USER IN A NATURAL WAY IN THE RESPONSE.
    """
    base_prompt = base_prompt.format(command=command)
    prompt: str = ""
    if context is None or len(context) == 0:
        prompt += str(base_prompt) + "context: NO CONTEXT AVAILABLE " + "question: " +  str(command.chatLog[-1])
    
    else:
        prompt += str(base_prompt) + "context: " + str(context[0])+ "question: "+ str(command.chatLog[-1])

    print(f"Prompt sent to LLM:\n{prompt}")

    language_model = create_llm(model)
    response = language_model.generate(prompt)
    
    return {
        "id": str(uuid.uuid4()),
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "logprobs": None,
                "finish_reason": "stop"
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
            "response_length": len(response),
            # "confidence_score": 0.95
        },
        "response": response
    }



