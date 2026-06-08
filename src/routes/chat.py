"""Chat and conversation endpoints for the RAGdoll service.

This module handles the main chat functionality including:
- Agent-based question answering with RAG
- Audio transcription to text
- Combined transcribe + answer workflows
"""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import Config
from src.globals import access_service, agent_dao
from src.models.chat.command import (
    Command,
    command_from_json_transcribe_version,
)
from src.models.errors import LLMAPIError, LLMGenerationError
from src.pipeline import assemble_prompt_with_agent
from src.routes.progress import get_recent_progress_for_session
from src.transcribe import transcribe_audio, transcribe_from_upload
from src.tts import get_tts_service
from src.whisper_model import warmup_whisper_model


router = APIRouter(prefix="/api/chat", tags=["chat"])
config = Config()


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str | None = None


class TTSWarmupRequest(BaseModel):
    language: str | None = None


class AskWithSpeechRequest(BaseModel):
    command: Command
    tts_language: str | None = None


def _get_authorized_agent(command: Command):
    agent = agent_dao.get_agent_by_id(command.agent_id)
    if agent is None:
        return None, JSONResponse(
            content={"message": f"Agent with id '{command.agent_id}' not found."},
            status_code=400,
        )

    if not access_service.authenticate(agent.id, command.access_key):
        raise HTTPException(status_code=401, detail="Unauthorized, check logs")

    if command.active_role_id and agent.get_role_by_name(command.active_role_id) is None:
        return None, JSONResponse(
            content={
                "message": f"Role '{command.active_role_id}' not found in agent '{agent.name}'."
            },
            status_code=400,
        )

    return agent, None


def _attach_recent_progress(command: Command) -> None:
    if not command.include_progress or not command.session_id or command.progress_limit <= 0:
        return

    recent_progress = get_recent_progress_for_session(
        command.agent_id, command.session_id, command.progress_limit
    )
    if not recent_progress:
        return

    explicit_task_names = {task.task_name for task in command.progress}
    merged_progress = [
        task for task in recent_progress if task.task_name not in explicit_task_names
    ]
    merged_progress.extend(command.progress)
    command.progress = merged_progress[: command.progress_limit]


def _generate_speech_payload(text: str, language: str | None = None) -> dict:
    return get_tts_service().synthesize(text, language).to_dict()


def _build_response_with_speech(response: dict, language: str | None = None) -> dict:
    return {
        "response": response,
        "speech": _generate_speech_payload(response.get("response", ""), language),
    }


@router.post("/ask", response_model=Command)
async def ask(command: Command):
    """Process a user question using the specified agent and roles.

    This endpoint:
    1. Receives a Command with agent_id and active_role_id
    2. Retrieves the agent configuration from the DAO
    3. Validates access permissions
    4. Performs RAG retrieval based on role-specific corpus access
    5. Generates a response using the agent's configured LLM

    Request body should be a JSON Command object with:
    - agent_id: MongoDB ObjectId of the agent
    - active_role_id: Name of the active agent role
    - access_key: Optional API key for authorization
    - chat_log: Conversation history
    - Other context fields (scene_name, user_information, etc.)

    Returns:
        JSONResponse with the generated answer and metadata

    Raises:
        400: Invalid command format, agent not found, or role not found
        401: LLM authentication failed (invalid API key or permissions)
        402: Insufficient LLM tokens/credits
        404: LLM model not found or not accessible
        429: LLM rate limit or quota exceeded
        503: LLM generation service error
        500: Other processing errors
    """
    try:
        agent, error_response = _get_authorized_agent(command)
        if error_response is not None:
            return error_response
        _attach_recent_progress(command)

        # Generate response using agent configuration and role-based RAG
        response = assemble_prompt_with_agent(command, agent)
        return JSONResponse(content={"response": response}, status_code=200)

    except LLMAPIError as e:
        # API-related errors (auth, quota, model not found, insufficient tokens)
        return JSONResponse(
            content={"message": str(e)},
            status_code=e.status_code,
        )
    except LLMGenerationError as e:
        # Generation/service errors
        return JSONResponse(
            content={"message": str(e)},
            status_code=e.status_code,
        )
    except UnicodeDecodeError:
        return JSONResponse(
            content={"message": "Invalid encoding. Expected UTF-8 encoded JSON."},
            status_code=400,
        )
    except HTTPException:
        raise
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


@router.post("/stt/warmup")
async def stt_warmup():
    """Lazy-load and warm the speech-to-text model."""
    try:
        result = warmup_whisper_model()
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": f"Failed to warm STT model: {e!s}"},
            status_code=500,
        )


@router.post("/tts/warmup")
async def tts_warmup(request: TTSWarmupRequest | None = None):
    """Lazy-load and warm the configured local text-to-speech model."""
    try:
        language = request.language if request else None
        result = get_tts_service().warmup(language)
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": f"Failed to warm TTS model: {e!s}"},
            status_code=500,
        )


@router.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Synthesize text using the configured local text-to-speech engine."""
    try:
        result = _generate_speech_payload(request.text, request.language)
        return JSONResponse(content={"speech": result}, status_code=200)
    except ValueError as e:
        return JSONResponse(content={"message": str(e)}, status_code=400)
    except FileNotFoundError as e:
        return JSONResponse(content={"message": str(e)}, status_code=503)
    except Exception as e:
        return JSONResponse(
            content={"message": f"Failed to synthesize speech: {e!s}"},
            status_code=500,
        )


@router.post("/askWithSpeech")
async def ask_with_speech(request: AskWithSpeechRequest):
    """Process a text chat request and return both text and local speech audio."""
    command = request.command

    try:
        agent, error_response = _get_authorized_agent(command)
        if error_response is not None:
            return error_response
        _attach_recent_progress(command)

        response = assemble_prompt_with_agent(command, agent)
        return JSONResponse(
            content=_build_response_with_speech(response, request.tts_language),
            status_code=200,
        )
    except LLMAPIError as e:
        return JSONResponse(content={"message": str(e)}, status_code=e.status_code)
    except LLMGenerationError as e:
        return JSONResponse(content={"message": str(e)}, status_code=e.status_code)
    except FileNotFoundError as e:
        return JSONResponse(content={"message": str(e)}, status_code=503)
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            content={"message": f"Error processing speech request: {e!s}"},
            status_code=500,
        )


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
    try:
        transcribed = transcribe_from_upload(audio)
    except ValueError as e:
        return JSONResponse(content={"message": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse(
            content={"message": f"Failed to transcribe audio: {e!s}"},
            status_code=500,
        )

    # Parse command and add transcribed text as user message
    command = command_from_json_transcribe_version(data, question=transcribed)
    if command is None:
        return JSONResponse(
            content={"message": "Invalid command format."}, status_code=400
        )

    agent, error_response = _get_authorized_agent(command)
    if error_response is not None:
        return error_response
    _attach_recent_progress(command)

    try:
        response = assemble_prompt_with_agent(command, agent)
        return JSONResponse(
            content={"transcription": transcribed, "response": response},
            status_code=200,
        )
    except LLMAPIError as e:
        return JSONResponse(content={"message": str(e)}, status_code=e.status_code)
    except LLMGenerationError as e:
        return JSONResponse(content={"message": str(e)}, status_code=e.status_code)


@router.post("/askTranscribeWithSpeech")
async def ask_transcribe_with_speech(
    audio: UploadFile = File(...),  # noqa: B008
    data: str = Form(...),
    tts_language: str = Form(None),
):
    """Transcribe audio, ask the agent, and return text plus local speech audio."""
    try:
        transcribed = transcribe_from_upload(audio)
    except ValueError as e:
        return JSONResponse(content={"message": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse(
            content={"message": f"Failed to transcribe audio: {e!s}"},
            status_code=500,
        )

    command = command_from_json_transcribe_version(data, question=transcribed)
    if command is None:
        return JSONResponse(
            content={"message": "Invalid command format."}, status_code=400
        )

    try:
        agent, error_response = _get_authorized_agent(command)
        if error_response is not None:
            return error_response
        _attach_recent_progress(command)

        response = assemble_prompt_with_agent(command, agent)
        return JSONResponse(
            content={
                "transcription": transcribed,
                **_build_response_with_speech(response, tts_language),
            },
            status_code=200,
        )
    except LLMAPIError as e:
        return JSONResponse(content={"message": str(e)}, status_code=e.status_code)
    except LLMGenerationError as e:
        return JSONResponse(content={"message": str(e)}, status_code=e.status_code)
    except FileNotFoundError as e:
        return JSONResponse(content={"message": str(e)}, status_code=503)
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            content={"message": f"Error processing speech request: {e!s}"},
            status_code=500,
        )
