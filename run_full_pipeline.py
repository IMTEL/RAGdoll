

import os

from src.pipeline import assemble_prompt_with_agent, generate_retrieval_query
from src.models.chat.command import Command
from src.routes.agents import create_agent, new_access_key
from src.models.agent import Agent, Role
from src.routes.api_keys import CreateAPIKeyRequest, create_api_key, get_api_key_detail


# Store the raw keys before creating UserAPIKey objects
llm_secret_key = os.getenv("IDUN_API_KEY")
gemini_secret_key = os.getenv("GEMINI_API_KEY")

llm_model_label = "model1"
llm_provider = "Idun"
llm_usage = "llm"

api_request = CreateAPIKeyRequest(
    label=llm_model_label,
    provider=llm_provider,
    usage=llm_usage,
    raw_key=llm_secret_key
)
api_key_llm_response = create_api_key(api_request)

print("Created API Key for LLM:", api_key_llm_response)
print("API Key ID:", api_key_llm_response.id)

# Get the full details with raw key
api_key_llm_detail = get_api_key_detail(api_key_llm_response.id)
print(f"Retrieved raw LLM key: {api_key_llm_detail.raw_key[:10]}...{api_key_llm_detail.raw_key[-4:]}")

api_request_embedding = CreateAPIKeyRequest(
    label="embedding_key",
    provider="gemini",
    usage="embedding",
    raw_key=gemini_secret_key
)
api_key_embedding_response = create_api_key(api_request_embedding)

print("Created API Key for Embedding:", api_key_embedding_response)
print("API Key ID:", api_key_embedding_response.id)

# Get the full details with raw key
api_key_embedding_detail = get_api_key_detail(api_key_embedding_response.id)
print(f"Retrieved raw Embedding key: {api_key_embedding_detail.raw_key[:10]}...{api_key_embedding_detail.raw_key[-4:]}")

role1 = Role(name="CrazyFrog", description="You are crazyfrog")
role2 = Role(name="Crazy frog", description="You are an clinically insane amphibian. You speak only in either frog language or overly sophisticated English.")

agent_body = Agent(
    name ="Test Agent",
    description="An agent for testing API key creation",
    prompt="You are a helpful assistant.",
    roles=[role1, role2],
    llm_provider=llm_provider,
    llm_model=os.getenv("IDUN_MODEL"),
    llm_temperature=0.7,
    llm_max_tokens=1000,
    llm_api_key=api_key_llm_detail.raw_key,  
    access_key=[],
    retrieval_method="semantic",
    embedding_model="models/text-embedding-004",
    status="active",
    response_format="text",
    last_updated="2024-06-01T12:00:00Z",
    embedding_api_key=api_key_embedding_detail.raw_key,  
    
    top_k=5,
    similarity_threshold=0.7,
    hybrid_search_alpha=0.75,   
)

agent = create_agent(agent_body)
print("Created Agent:", agent)

access_key = new_access_key(name="access_key1", agent_id=agent.id)

agent.access_key.append(access_key)


command = Command(
    chat_log=[
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "what are flamingos?"},
    ],
    agent_id=agent.id,
    active_role_id="CrazyFrog",
)


response = assemble_prompt_with_agent(command, agent)

print("Generated Response:")
print(response["response"])





