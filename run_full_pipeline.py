

import os

from src.routes.agents import create_agent, new_access_key
from src.models.agent import Agent, Role
from src.routes.api_keys import CreateAPIKeyRequest, create_api_key


llm_secret_key = os.getenv("IDUN_API_KEY")
llm_model_label = "model1"
llm_provider = "Idun"
llm_usage = "llm"

api_request = CreateAPIKeyRequest(
    label=llm_model_label,
    provider=llm_provider,
    usage=llm_usage,
    raw_key=llm_secret_key
)
api_key_llm = create_api_key(api_request)

print("Created API Key for LLM:", api_key_llm)
print("API Key ID:", api_key_llm.id)

api_request_embedding = CreateAPIKeyRequest(
    label="embedding_key",
    provider="gemini",
    usage="embedding",
    raw_key=os.getenv("GEMINI_API_KEY")
)
api_key_embedding = create_api_key(api_request_embedding)

print("Created API Key for Embedding:", api_key_embedding)
print("API Key ID:", api_key_embedding.id)

role1 = Role(name="CrazyFrog", description="You are crazyfrog")
role2 = Role(name="Crazy fog", description="You are an clinically insane amphibian. You speak only in either frog language or overly sophisticated English.")

agent_body = Agent(
    name ="Test Agent",
    description="An agent for testing API key creation",
    prompt="You are a helpful assistant.",
    roles=[role1, role2],
    llm_provider=llm_provider,
    llm_model=llm_model_label,
    llm_temperature=0.7,
    llm_max_tokens=1000,
    llm_api_key=api_key_llm.redacted_key,
    access_key=[],
    retrieval_method="semantic",
    embedding_model="gemini-2.0-flash-lite",
    status="active",
    response_format="text",
    last_updated="2024-06-01T12:00:00Z",
    embedding_api_key=api_key_embedding.redacted_key,
    
    top_k=5,
    similarity_threshold=0.7,
    hybrid_search_alpha=0.75,   
)

agent = create_agent(agent_body)
print("Created Agent:", agent)

access_key = new_access_key(name="access_key1", agent_id=agent.id)

agent.access_key.append(access_key)

