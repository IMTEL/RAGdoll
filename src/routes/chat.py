"""Chat and conversation endpoints for the RAGdoll service.

This module handles the main chat functionality including:
- Agent-based question answering with RAG
- Audio transcription to text
- Combined transcribe + answer workflows
"""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.access_service.factory import AccessServiceConfig, access_service_factory
from src.config import Config
from src.models.chat.command import (
    Command,
    command_from_json_transcribe_version,
)
from src.pipeline import assemble_prompt_with_agent
from src.rag_service.dao import get_agent_dao
from src.transcribe import transcribe_audio, transcribe_from_upload


router = APIRouter(prefix="/api/chat", tags=["chat"])
config = Config()
agent_dao = get_agent_dao()

access_service = access_service_factory(
    AccessServiceConfig(config.ACCESS_SERVICE, agent_dao)
)


@router.post("/ask", response_model=Command)
async def ask(command: Command):
    """Process a user question using the specified agent and roles.

    This endpoint:
    1. Receives a Command with agent_id and active_role_id
    2. Retrieves the agent configuration from the DAO
    3. Validates access permissions
    4. Performs RAG retrieval based on role-specific corpus access
    5. Generates a response using the agent's LLM configuration

    Request body should be a JSON Command object with:
    - agent_id: MongoDB ObjectId of the agent
    - active_role_id: Name of the active agent role
    - access_key: Optional API key for authorization
    - chat_log: Conversation history
    - Other context fields (scene_name, user_information, etc.)

    Returns:
        JSONResponse with the generated answer

    Raises:
        400: Invalid command format, agent not found, or access denied
        500: Processing error
    """
    try:
        # Retrieve the agent configuration
        agent = agent_dao.get_agent_by_id(command.agent_id)
        if agent is None:
            return JSONResponse(
                content={"message": f"Agent with id '{command.agent_id}' not found."},
                status_code=400,
            )

        # Auth
        if not access_service.authenticate(agent.id, command.access_key):
            raise HTTPException(status_code=401, details="Unauthorized, check logs")

        # Validate that requested role exists in the agent
        if (
            command.active_role_id
            and agent.get_role_by_name(command.active_role_id) is None
        ):
            return JSONResponse(
                content={
                    "message": f"Role '{command.active_role_id}' not found in agent '{agent.name}'."
                },
                status_code=400,
            )

        # Generate response using agent configuration and role-based RAG
        response = assemble_prompt_with_agent(command, agent)
        return JSONResponse(content={"response": response}, status_code=200)

    except UnicodeDecodeError:
        return JSONResponse(
            content={"message": "Invalid encoding. Expected UTF-8 encoded JSON."},
            status_code=400,
        )
    except Exception as e:
        return JSONResponse(
            content={"message": f"Error processing request: {e!s}"}, status_code=500
        )


@router.post("/transcribe")
async def transcribe_endpoint(
    audio: UploadFile = File(...),  # noqa: B008
    language: str = Form(None),
):
    """Transcribe an audio file to text using Whisper.

    Parameters:
    - audio: The audio file (WAV format recommended)
    - language: Optional language code (e.g., 'en', 'fr', 'es')

    Returns:
    - A JSON response with transcription or error message
    """
    result = transcribe_audio(audio, language)

    if result["success"]:
        return JSONResponse(content=result, status_code=200)
    else:
        return JSONResponse(content=result, status_code=400)


@router.post("/askTranscribe")
async def ask_transcribe(
    audio: UploadFile = File(...),  # noqa: B008
    data: str = Form(...),
):
    """Transcribe audio and process with agent in a single request.

    This combines audio transcription with the ask endpoint:
    1. Transcribes the audio file to text
    2. Uses that text as the user's question
    3. Processes the question with the specified agent

    Parameters:
    - audio: The audio file containing the user's question
    - data: JSON string with Command data (except chat_log which is generated)

    Returns:
    - A JSON response with the agent's answer
    """
    # Transcribe the audio
    transcribed = transcribe_from_upload(audio)

    # Parse command and add transcribed text as user message
    command = command_from_json_transcribe_version(data, question=transcribed)
    if command is None:
        return JSONResponse(
            content={"message": "Invalid command format."}, status_code=400
        )

    # Retrieve and validate agent (same logic as /ask)
    agent = agent_dao.get_agent_by_id(command.agent_id)
    if agent is None:
        return JSONResponse(
            content={"message": f"Agent with id '{command.agent_id}' not found."},
            status_code=400,
        )

    # Generate response
    response = assemble_prompt_with_agent(command, agent)
    return JSONResponse(
        content={"transcription": transcribed, "response": response}, status_code=200
    )
