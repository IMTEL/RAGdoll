
from src.command import Command, Prompt, prompt_to_json
from src.rag_service.dao import get_database
from src.config import Config
from src.rag_service.embeddings import create_embeddings_model
from src.LLM import LLM, create_llm

def assemble_prompt(command: Command) -> str:
    """Assembles a prompt for a large language model and prompt LLM to generate a response."""
    # TODO: use rag service to get the context, and create a Prompt pydantic object.
    
    
    to_embed = " ".join([command.question] + command.progress + command.user_actions)
    
    
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
    if context is None:
        prompt = base_prompt, "question:", command.question
    
    else:
        prompt = base_prompt, "context:", context[0], "question:", command.question
    
    language_model = create_llm()
    response = language_model.generate(prompt)
    
    return response



