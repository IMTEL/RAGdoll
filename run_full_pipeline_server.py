"""
Model-Driven Agent Server

This script runs the full RAGdoll pipeline and exposes it as a local API server.
When the frontend has USE_MODEL_DRIVEN_AGENT=true, it connects to this server
instead of the main API, allowing for direct pipeline interaction.

This enables:
- Rapid prototyping without deploying to main API
- Testing new agent configurations locally
- Direct pipeline debugging
- Isolated agent development
"""

import os
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from src.pipeline import assemble_prompt_with_agent
from src.models.chat.command import Command
from src.models.chat.message import Message
from src.routes.agents import create_agent, new_access_key
from src.models.agent import Agent, Role
from src.routes.api_keys import CreateAPIKeyRequest, create_api_key, get_api_key_detail

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global agent storage
PIPELINE_AGENT: Optional[Agent] = None
AGENT_CONFIG = {}


def initialize_pipeline_agent():
    """Initialize the agent that will be used in the pipeline."""
    global PIPELINE_AGENT, AGENT_CONFIG
    
    logger.info("=" * 80)
    logger.info("Initializing Pipeline Agent")
    logger.info("=" * 80)
    

    
    llm_secret_key = os.getenv("IDUN_API_KEY")
    if not llm_secret_key:
        raise ValueError("IDUN_API_KEY environment variable not set")
    
    gemini_secret_key = os.getenv("GEMINI_API_KEY")
    if not gemini_secret_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    # Create LLM API key
    api_request_llm = CreateAPIKeyRequest(
        label="pipeline_llm_key",
        provider="idun",
        usage="llm",
        raw_key=llm_secret_key
    )
    api_key_llm_response = create_api_key(api_request_llm)
    api_key_llm_detail = get_api_key_detail(api_key_llm_response.id)
    
    api_request_embedding = CreateAPIKeyRequest(
        label="pipeline_embedding_key",
        provider="gemini",
        usage="embedding",
        raw_key=gemini_secret_key
    )
    api_key_embedding_response = create_api_key(api_request_embedding)
    api_key_embedding_detail = get_api_key_detail(api_key_embedding_response.id)
    logger.info(f"Created Embedding API key: {api_key_embedding_response.id}")
    
    
    roles = [
        Role(
            name="CrazyFrog",
            description="You are crazyfrog. You act like CrazyFrog from the internet memes.",
            document_access=["trex.pdf"]
        ),
        Role(
            name="Crazy frog",
            description="You are an clinically insane amphibian. You speak only in either frog language or overly sophisticated English.",
            document_access=["trex.pdf"]
        ),
        Role(
            name="Crazy Zebra",
            description="You are a zebra with a wild imagination. You often confuse reality with fantasy.",
            document_access=["trex.pdf"]
        )
    ]
    logger.info(f"   ✓ Created {len(roles)} roles")
    
    # Step 3: Create Agent
    logger.info("📝 Step 3: Creating agent...")
    
    agent_body = Agent(
        id="6932698c69151632bf86532f",  # Hardcoded agent ID to match uploaded documents
        name="Pipeline Test Agent",
        description="An agent for testing the model-driven pipeline",
        prompt=(
            "Fuck around and find out."
        ),
        roles=roles,
        llm_provider="idun",
        llm_model=os.getenv("IDUN_MODEL", "openai/gpt-oss-120b"),
        llm_temperature=0.7,
        llm_max_tokens=1000,
        llm_api_key=api_key_llm_detail.raw_key,
        access_key=[],
        retrieval_method="hybrid",
        embedding_model="gemini:models/text-embedding-004",
        status="active",
        response_format="text",
        last_updated="2024-06-01T12:00:00Z",
        embedding_api_key=api_key_embedding_detail.raw_key,
        top_k=5,
        similarity_threshold=0.7,
        hybrid_search_alpha=0.75,
    )
    
    agent = create_agent(agent_body)
    logger.info(f"Created agent: {agent.name} (ID: {agent.id})")
    
    # Create access key
    access_key = new_access_key(name="pipeline_access_key", agent_id=agent.id)
    agent.access_key.append(access_key)
    logger.info(f"Created access key")
    
    PIPELINE_AGENT = agent
    AGENT_CONFIG = {
        "id": agent.id,
        "name": agent.name,
        "description": agent.description,
        "roles": [{"name": r.name, "description": r.description} for r in agent.roles],
        "llm_provider": agent.llm_provider,
        "llm_model": agent.llm_model,
        "embedding_model": agent.embedding_model,
        "top_k": agent.top_k,
        "similarity_threshold": agent.similarity_threshold,
    }
    
    logger.info("=" * 80)
    logger.info(f"Agent ID: {agent.id}")
    logger.info(f"Agent Name: {agent.name}")
    logger.info(f"Roles: {', '.join([r.name for r in agent.roles])}")
    logger.info("=" * 80)
    
    return agent


# FastAPI lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agent on startup, cleanup on shutdown."""
    logger.info("Starting Model-Driven Agent Server")
    
    # Startup
    try:
        initialize_pipeline_agent()
    except Exception as e:
        logger.error(f"Failed to initialize pipeline agent: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Model-Driven Agent Server")


# Create FastAPI app
app = FastAPI(
    title="Model-Driven Agent Server",
    description="Local pipeline server for model-driven agent interaction",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for requests/responses
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    active_role_id: str = "assistant"


class ChatResponse(BaseModel):
    response: str
    agent_id: str
    agent_name: str
    role: str
    contexts_used: int
    contexts: list[dict]


# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "mode": "model-driven",
        "agent": AGENT_CONFIG if PIPELINE_AGENT else None
    }


@app.get("/api/agent")
async def get_agent():
    """Get the initialized agent configuration."""
    if not PIPELINE_AGENT:
        raise HTTPException(status_code=503, detail="Pipeline agent not initialized")
    
    return {
        "id": PIPELINE_AGENT.id,
        "name": PIPELINE_AGENT.name,
        "description": PIPELINE_AGENT.description,
        "prompt": PIPELINE_AGENT.prompt,
        "roles": [
            {
                "name": r.name,
                "description": r.description,
                "document_access": r.document_access
            }
            for r in PIPELINE_AGENT.roles
        ],
        "llm_provider": PIPELINE_AGENT.llm_provider,
        "llm_model": PIPELINE_AGENT.llm_model,
        "embedding_model": PIPELINE_AGENT.embedding_model,
        "status": PIPELINE_AGENT.status,
        "access_key": PIPELINE_AGENT.access_key[0] if PIPELINE_AGENT.access_key else None,
    }


@app.get("/api/agents")
async def list_agents():
    """List all available agents (in this case, just the pipeline agent)."""
    if not PIPELINE_AGENT:
        return {"agents": []}
    
    return {
        "agents": [AGENT_CONFIG]
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - processes messages through the pipeline.
    
    This endpoint:
    1. Receives chat messages from the frontend
    2. Constructs a Command object
    3. Runs it through assemble_prompt_with_agent
    4. Returns the response with context information
    """
    if not PIPELINE_AGENT:
        raise HTTPException(status_code=503, detail="Pipeline agent not initialized")
    
    
    try:
        chat_log = [
            Message(role=msg.role, content=msg.content)
            for msg in request.messages
        ]
        
        command = Command(
            chat_log=chat_log,
            agent_id=PIPELINE_AGENT.id,
            active_role_id=request.active_role_id,
        )
        
        result = assemble_prompt_with_agent(command, PIPELINE_AGENT)
        
        contexts = []
        if result.get("context_used"):
            contexts = [
                {
                    "document_name": ctx["document_name"],
                    "chunk_index": ctx["chunk_index"],
                    "content": ctx["content"]
                }
                for ctx in result["context_used"]
            ]
        
        logger.info(f"   Response generated ({len(result['response'])} chars)")
        logger.info(f"   Contexts used: {len(contexts)}")
        
        return ChatResponse(
            response=result["response"],
            agent_id=PIPELINE_AGENT.id,
            agent_name=PIPELINE_AGENT.name,
            role=request.active_role_id,
            contexts_used=len(contexts),
            contexts=contexts
        )
        
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/chat/{agent_id}")
async def websocket_chat(websocket: WebSocket, agent_id: str):
    """
    WebSocket endpoint for real-time chat.
    
    This enables streaming responses and maintains a persistent connection.
    """
    await websocket.accept()
    
    if not PIPELINE_AGENT or agent_id != PIPELINE_AGENT.id:
        await websocket.send_json({
            "error": "Agent not found or not initialized"
        })
        await websocket.close()
        return
    
    logger.info(f"WebSocket connection established for agent {agent_id}")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            messages = data.get("messages", [])
            active_role = data.get("active_role_id", "assistant")
            
            # Convert to Message objects
            chat_log = [
                Message(role=msg["role"], content=msg["content"])
                for msg in messages
            ]
            
            # Create command
            command = Command(
                chat_log=chat_log,
                agent_id=PIPELINE_AGENT.id,
                active_role_id=active_role,
            )
            
            # Run through pipeline
            result = assemble_prompt_with_agent(command, PIPELINE_AGENT)
            
            # Send response back
            await websocket.send_json({
                "type": "response",
                "response": result["response"],
                "contexts_used": result["metadata"]["num_context_retrieved"],
                "agent_name": PIPELINE_AGENT.name
            })
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for agent {agent_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({"error": str(e)})
        await websocket.close()


def main():
    """Run the server."""
    port = int(os.getenv("MODEL_DRIVEN_PORT", "8001"))
    host = os.getenv("MODEL_DRIVEN_HOST", "0.0.0.0")
    
    logger.info("=" * 80)
    logger.info("Starting Model-Driven Agent Server")
    logger.info("=" * 80)
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"Endpoints:")
    logger.info(f"  - GET  /                      - Health check")
    logger.info(f"  - GET  /api/agent/{{agent_id}}  - Get agent config")
    logger.info(f"  - GET  /api/agents            - List agents")
    logger.info(f"  - POST /api/chat              - Chat with agent")
    logger.info(f"  - WS   /ws/chat/{{agent_id}}   - WebSocket chat")
    logger.info("=" * 80)
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
