from fastapi import APIRouter

from src.models.agent import Agent, AgentRead
from src.rag_service.agent_dao import get_agent_database


router = APIRouter()
agent_db = get_agent_database()  # your DAO


# Serialize MongoDB _id → id
def serialize_agent(agent_doc):
    agent_doc["id"] = str(agent_doc["_id"])
    agent_doc.pop("_id", None)
    return agent_doc


# Create a new agent
@router.post("/agents/", response_model=Agent)
def create_agent(agent: Agent):
    # Let DAO handle insertion
    return agent_db.create_agent(agent)


# Get all agents
@router.get("/agents/", response_model=list[AgentRead])
def get_agents():
    agents = agent_db.get_agents()
    return [serialize_agent(agent) for agent in agents]
