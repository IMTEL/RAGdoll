
from src.command import Command, Prompt, prompt_to_json
from src.rag_service.dao import get_database
from src.config import Config
from src.rag_service.embeddings import create_embeddings_model
from src.LLM import LLM, create_llm

def assemble_prompt(command: Command) -> str:
    """Assembles a prompt for a large language model and prompt LLM to generate a response."""
    # TODO: use rag service to get the context, and create a Prompt pydantic object.
    
    #to_embed: str = str(command.question) + " "+ str(command.progress) + " "+ str(command.user_actions)
    to_embed: str = str(command.question)
    
    db = get_database()
    embedding_model = create_embeddings_model()
    embeddings: list[float] = embedding_model.get_embedding(to_embed)
    context = db.get_context("hello", embeddings ) # TODO: remove document name from here
    
    base_prompt = """
    You are a helpful assistant and guide in the Blue Sector Virtual Reality work training. 
    You are here to help the user with their questions and guide them through the training.
    The name of the user is {user_name}.
    The user is in {user_mode} mode.
    The user has made the following progress: {progress}.
    The user has taken the following actions: {user_actions}.
    """
    prompt: str = ""
    if context is None or len(context) == 0:
        prompt += str(base_prompt) + "question:" +  str(command.question)
        print(context[0])
    
    else:
        prompt += str(base_prompt) + "context:" + str(context[0])+ "question:"+ str(command.question)
    
    language_model = create_llm()
    response = language_model.generate(prompt)
    
    return response



