"""Chat and conversation endpoints for the RAGdoll service.

This module handles the main chat functionality including:
- Agent-based question answering with RAG
- Audio transcription to text
- Combined transcribe + answer workflows
"""

from typing import Annotated

from fastapi import (
    APIRouter,
    Body,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from src.models.agent import Agent

from src.config import Config
from src.models.chat.command import (
    Command,
    command_from_json_transcribe_version,
)
from src.models.errors import LLMAPIError, LLMGenerationError
from src.pipeline import assemble_prompt_with_agent
from src.routes.progress import get_recent_progress_for_session
from src.transcribe import transcribe_audio
from src.tts import get_tts_service
from src.utils.path_dependencies import get_auth, has_agent_call_access
from src.whisper_model import warmup_whisper_model


router = APIRouter(prefix="/api/chat", tags=["chat"])
config = Config()


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str | None = None


class TTSWarmupRequest(BaseModel):
    language: str | None = None


class STTWarmupRequest(BaseModel):
    language: str | None = None


class AskWithSpeechRequest(BaseModel):
    command: Command
    tts_language: str | None = None


def _attach_recent_progress(command: Command) -> None:
    if (
        not command.include_progress
        or not command.session_id
        or command.progress_limit <= 0
    ):
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


@router.post(
    "/ask",
    responses={
        404: {"description": "If a role could not be found within the agent"},
        500: {"description": "If the server failed to process the request"},
        400: {"description": "If the encoding is something else then utf-8"},
    },
)
async def ask(command: Command, request: Request):
    # TODO: We need a unviersal return object for LLMs. I have no idea about the current properties, which is why I am setting this dict until further notice
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
    command
        Raises:
            400: Invalid command format, agent not found, or role not found
            401: LLM authentication failed (invalid API key or permissions)
            402: Insufficient LLM tokens/credits
            404: LLM model not found or not accessible
            429: LLM rate limit or quota exceeded
            503: LLM generation service error
            500: Other processing errors
    """
    agent: Agent
    # Until further notice we accept both the authorization header and the command access_key
    # Someone else can remove it down the line
    if command.access_key == None:
        agent = has_agent_call_access(command.agent_id, get_auth(request))
    else:
        agent = has_agent_call_access(command.agent_id, command.access_key)
    if (
        command.active_role_id
        and agent.get_role_by_name(command.active_role_id) is None
    ):
        raise HTTPException(status_code=404, detail="Role not found in agent")
    try:
        _attach_recent_progress(command)

        # Generate response using agent configuration and role-based RAG
        response = assemble_prompt_with_agent(command, agent)
        return response

    except LLMAPIError as e:
        # API-related errors (auth, quota, model not found, insufficient tokens)
        raise HTTPException(
            detail=str(e),
            status_code=e.status_code,
        ) from e
    except LLMGenerationError as e:
        # Generation/service errors
        raise HTTPException(
            detail={str(e)},
            status_code=e.status_code,
        ) from e
    except UnicodeDecodeError as e:
        raise HTTPException(
            detail="Invalid encoding. Expected UTF-8 encoded JSON.",
            status_code=400,
        ) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            detail=f"Error processing request: {e!s}", status_code=500
        ) from e


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
async def stt_warmup(request: STTWarmupRequest | None = None):
    """Lazy-load and warm the speech-to-text model."""
    try:
        result = warmup_whisper_model(request.language if request else None)
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
async def ask_with_speech(request: AskWithSpeechRequest, request_object: Request):
    """Process a text chat request and return both text and local speech audio."""
    response = await ask(request.command, request_object)
    try:
        return JSONResponse(
            content=_build_response_with_speech(response, request.tts_language),
            status_code=200,
        )
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
    request_object: Request,
    audio: Annotated[UploadFile, File()],
    data: Annotated[str, Body()] = Form(...),
    stt_language: Annotated[str, Body()] = Form(None),
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
    transcription_result = transcribe_audio(audio, stt_language)
    if not transcription_result["success"]:
        raise HTTPException(
            detail=transcription_result["error"],
            status_code=400,
        )
    transcribed = transcription_result["transcription"]

    # Parse command and add transcribed text as user message
    # TODO: This is one of the many reasons we have pydmantic. Let it do the casting
    command = command_from_json_transcribe_version(data, question=transcribed)
    if command is None:
        raise HTTPException(status_code=400, detail="Invalid command format")
    response = await ask(command, request_object)
    return {
        "transcription": transcribed,
        "transcription_language": transcription_result["language"],
        "transcription_language_probability": transcription_result[
            "language_probability"
        ],
        "response": response,
    }


@router.post("/askTranscribeWithSpeech")
async def ask_transcribe_with_speech(
    request_object: Request,
    audio: UploadFile = File(...),  # noqa: B008
    data: str = Form(...),
    tts_language: str = Form(None),
    stt_language: str = Form(None),
):
    """Transcribe audio, ask the agent, and return text plus local speech audio."""
    response = await ask_transcribe(request_object, audio, data, stt_language)
    try:
        return (
            {
                **response,
                **_build_response_with_speech(response, tts_language),
            },
        )
    except FileNotFoundError as e:
        raise HTTPException(detail={"message": str(e)}, status_code=503) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            detail={f"Error processing speech request: {e!s}"},
            status_code=500,
        ) from e
