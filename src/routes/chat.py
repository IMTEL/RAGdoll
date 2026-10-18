"""Chat and conversation endpoints for the RAGdoll service.

This module handles the main chat functionality including:
- Agent-based question answering with RAG
- Audio transcription to text
- Combined transcribe + answer workflows
"""

import logging
from typing import TYPE_CHECKING

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from pymongo.errors import PyMongoError

from src.config import Config
from src.models.chat.command import (
    Command,
    command_from_json_transcribe_version,
)
from src.models.errors import (
    EmbeddingAPIError,
    LLMAPIError,
    LLMGenerationError,
    RetrievalFailedError,
)
from src.pipeline import assemble_prompt_with_agent
from src.routes.progress import get_recent_progress_for_session
from src.transcribe import transcribe_audio
from src.tts import get_tts_service, synthesize_sentences
from src.utils.path_dependencies import (
    get_bearer_token,
    get_optional_user,
    has_agent_call_access,
    require_speech_access,
)
from src.whisper_model import warmup_whisper_model


if TYPE_CHECKING:
    from src.models.agent import Agent


# Every handler here is a plain ``def`` on purpose: they call blocking code (LLM
# and embedding HTTP, Mongo, FAISS, Whisper, Piper), and FastAPI runs ``def``
# handlers in its threadpool. An ``async def`` handler would run that work on
# the event loop and stall every other request until it finished.
router = APIRouter(prefix="/api/chat", tags=["chat"])
config = Config()
logger = logging.getLogger(__name__)


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

    try:
        recent_progress = get_recent_progress_for_session(
            command.agent_id, command.session_id, command.progress_limit
        )
    except (PyMongoError, ValidationError) as e:
        # Progress is auxiliary context: answer without it rather than fail
        # the chat. Only the type is logged (driver errors can hold the URI).
        logger.warning(
            "Could not load recent progress for the chat (%s); answering without it",
            type(e).__name__,
        )
        return
    if not recent_progress:
        return

    explicit_task_names = {task.task_name for task in command.progress}
    merged_progress = [
        task for task in recent_progress if task.task_name not in explicit_task_names
    ]
    merged_progress.extend(command.progress)
    command.progress = merged_progress[: command.progress_limit]


def _generate_speech_payload(text: str, language: str | None = None) -> dict:
    """Synthesize sentences in parallel: ``speech`` (concatenated) + segments."""
    speech, segments = synthesize_sentences(get_tts_service(), text, language)
    return {"speech": speech, "speech_segments": segments}


def _build_response_with_speech(response: dict, language: str | None = None) -> dict:
    return {
        "response": response,
        **_generate_speech_payload(response.get("response", ""), language),
    }


@router.post(
    "/ask",
    responses={
        404: {"description": "If a role could not be found within the agent"},
        500: {"description": "If the server failed to process the request"},
        400: {"description": "If the encoding is something else then utf-8"},
    },
)
def ask(command: Command, request: Request):
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
            401: LLM or embedding authentication failed (invalid API key or
                 permissions, or a stored key reference that is no longer valid)
            402: Insufficient LLM or embedding tokens/credits
            404: LLM model not found or not accessible
            429: LLM or embedding rate limit or quota exceeded
            503: LLM generation service error, or knowledge retrieval failed
                 and the agent's on_retrieval_error is "fail"
            500: Other processing errors
    """
    agent = _authorize_command(command, request)
    return _answer(command, agent)


def _authorize_command(command: Command, request: Request) -> "Agent":
    """Returns the command's agent if the caller may call it.

    Raises:
        HTTPException: 401/403/404 from the bearer or access key checks
    """
    # Until further notice we accept both the authorization header and the command access_key
    # Someone else can remove it down the line
    credentials = (
        get_bearer_token(request) if command.access_key is None else command.access_key
    )
    return has_agent_call_access(command.agent_id, get_optional_user(credentials))


def _answer(command: Command, agent: "Agent") -> dict:
    """Answers an already authorized command with the agent's LLM."""
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

    except RetrievalFailedError as e:
        raise HTTPException(
            detail="Knowledge retrieval failed",
            status_code=503,
        ) from e
    except (LLMAPIError, EmbeddingAPIError) as e:
        # API-related errors (auth, quota, model not found, insufficient tokens)
        raise HTTPException(
            detail=str(e),
            status_code=e.status_code,
        ) from e
    except LLMGenerationError as e:
        # Generation/service errors
        raise HTTPException(
            detail=str(e),
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


@router.post("/transcribe", dependencies=[Depends(require_speech_access)])
def transcribe_endpoint(
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


@router.post("/stt/warmup", dependencies=[Depends(require_speech_access)])
def stt_warmup(request: STTWarmupRequest | None = None):
    """Lazy-load and warm the speech-to-text model."""
    try:
        result = warmup_whisper_model(request.language if request else None)
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": f"Failed to warm STT model: {e!s}"},
            status_code=500,
        )


@router.post("/tts/warmup", dependencies=[Depends(require_speech_access)])
def tts_warmup(request: TTSWarmupRequest | None = None):
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


@router.post("/tts", dependencies=[Depends(require_speech_access)])
def text_to_speech(request: TTSRequest):
    """Synthesize text using the configured local text-to-speech engine."""
    try:
        return JSONResponse(
            content=_generate_speech_payload(request.text, request.language),
            status_code=200,
        )
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
def ask_with_speech(request: AskWithSpeechRequest, request_object: Request):
    """Process a text chat request and return both text and local speech audio."""
    response = ask(request.command, request_object)
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
def ask_transcribe(
    request_object: Request,
    audio: UploadFile = File(...),  # noqa: B008
    data: str = Form(...),
    stt_language: str = Form(None),
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
    # Authenticate before transcribing, so anonymous callers never reach Whisper.
    # TODO: This is one of the many reasons we have pydmantic. Let it do the casting
    command = command_from_json_transcribe_version(data)
    if command is None:
        raise HTTPException(status_code=400, detail="Invalid command format")
    agent = _authorize_command(command, request_object)

    transcription_result = transcribe_audio(audio, stt_language)
    if not transcription_result["success"]:
        raise HTTPException(
            detail=transcription_result["error"],
            status_code=400,
        )
    transcribed = transcription_result["transcription"]

    # Parse the command again with the transcribed text as the user message
    command = command_from_json_transcribe_version(data, question=transcribed)
    response = _answer(command, agent)
    return {
        "transcription": transcribed,
        "transcription_language": transcription_result["language"],
        "transcription_language_probability": transcription_result[
            "language_probability"
        ],
        "response": response,
    }


@router.post("/askTranscribeWithSpeech")
def ask_transcribe_with_speech(
    request_object: Request,
    audio: UploadFile = File(...),  # noqa: B008
    data: str = Form(...),
    tts_language: str = Form(None),
    stt_language: str = Form(None),
):
    """Transcribe audio, ask the agent, and return text plus local speech audio."""
    response = ask_transcribe(request_object, audio, data, stt_language)
    try:
        answer = response["response"]
        answer_text = answer.get("response", "") if isinstance(answer, dict) else ""
        return {
            **response,
            **_generate_speech_payload(answer_text, tts_language),
        }
    except FileNotFoundError as e:
        raise HTTPException(detail=str(e), status_code=503) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            detail=f"Error processing speech request: {e!s}",
            status_code=500,
        ) from e
