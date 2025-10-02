from datetime import datetime
from fastapi import APIRouter, HTTPException
from typing import List

from src.auth_service.access_service import AccessServiceConfig, access_service_factory
from src.config import Config
from src.models.accesstoken import AccessKey
from ..models.agent import Agent, AgentRead
from ..rag_service.agent_dao import get_agent_database
from bson import ObjectId

config = Config()
router = APIRouter()
agent_db = get_agent_database()  # your DAO
access_service = access_service_factory(AccessServiceConfig(config.ACCESS_SERVICE,agent_db))

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
@router.get("/agents/", response_model=List[AgentRead])
def get_agents():
    agents = agent_db.get_agents()
    return [serialize_agent(agent) for agent in agents]

@router.post("/agents/new-accesskey", response_model=AccessKey)
def new_access_key(name: str, expiery_date : datetime, agent_id : str):
    try:
        return access_service.generate_accesstoken(name,expiery_date,agent_id)
    except e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error : {e}")


@router.get("/agents/revoke-accesskey")
def revoke_access_key(access_key: str):
    try:
        return access_service.revoke_key(access_key)
    except e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error : {e}")
    

