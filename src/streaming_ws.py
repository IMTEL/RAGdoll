# pip install websocket

#  src/streaming_ws.py
"""streaming_ws.py
-----------------
Adds full‑duplex WebSocket streaming to the chat‑service for:
    •  client → server   : raw PCM audio (+ control JSON frames)
    •  server → client   : partial transcripts, final transcript, token‑stream from the LLM, (future) TTS audio.

The module is 100 % self‑contained: simply `import` and `include_router(router)` in **src/main.py** and rebuild
Docker to enable streaming.

Protocol (all messages share the same WebSocket):
─────────────────────────────────────────────────
bytes      – 16 kHz, 16‑bit little‑endian mono PCM audio
text/JSON  – {"type": "<event>", ...}
    client→server events
        • silence            – VR client detected end‑of‑utterance
        • command            – Full serialized `Command` object for RAG/LLM
    server→client events
        • transcript_partial – incremental Whisper output
        • transcript_final   – final Whisper output (after `silence`)
        • llm_token          – single token from the LLM stream
        • error              – unexpected problems
        • tts_chunk          – *(future)* base64/wav audio matching the `llm_token`

Unity clients can keep a single socket open per NPC and multiplex speech turns with
simple counters inside the JSON payload.
"""

import asyncio
import json
import time
from collections import defaultdict
from typing import AsyncGenerator, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import io # Added for byte buffer handling

# Import the optimized model and settings from transcribe.py
from src.transcribe import model, TARGET_SR, DEFAULT_BEAM_SIZE, DEFAULT_BEST_OF, USE_EFFICIENT_BY_DEFAULT
from src.command import command_from_json, Command
from src.pipeline import assemble_prompt
from src.config import Config
from src.LLM import create_llm

# ──────────────────────────────────────────────────────────────────────────
# WebSocket endpoint
# ──────────────────────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active:
            self.active.remove(websocket)

manager = ConnectionManager()
router  = APIRouter()

@router.websocket("/ws/chat")
async def chat_stream(websocket: WebSocket):
    """Main bidirectional stream used by Unity clients."""

    await manager.connect(websocket)
    # Use a byte buffer to accumulate audio data
    audio_buffer = io.BytesIO() 
    pending_command_json: Optional[str] = None

    try:
        while True:
            message = await websocket.receive()

            # 1️  raw audio bytes
            if message["type"] == "websocket.receive" and "bytes" in message and message["bytes"]:
                # Append incoming audio bytes to the buffer
                audio_buffer.write(message["bytes"])
                
                # Check if we have enough audio data for streaming transcription
                # (Process every ~1 second of audio while recording)
                audio_buffer_size = audio_buffer.tell()
                if audio_buffer_size >= 32000:  # ~1 second of 16kHz 16-bit audio
                    # Process the current buffer without clearing it
                    try:
                        # Make a copy of current audio data for processing
                        current_position = audio_buffer.tell()
                        audio_buffer.seek(0)
                        audio_data_copy = audio_buffer.read(current_position)
                        audio_buffer.seek(current_position)  # Reset position after reading
                        
                        # Convert to numpy array for processing
                        audio_np = np.frombuffer(audio_data_copy, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        # Use the optimized model settings for faster transcription
                        segments, info = model.transcribe(
                            audio_np, 
                            beam_size=DEFAULT_BEAM_SIZE,  # Use faster beam size
                            language="en",
                            condition_on_previous_text=True,
                            vad_filter=True,
                            vad_parameters=dict(min_silence_duration_ms=300)  # Reduced for faster processing
                        )
                        
                        # Get partial transcript
                        partial_text = ""
                        segments_list = list(segments)  # Convert generator to list
                        if segments_list:
                            partial_text = "".join(segment.text for segment in segments_list)
                            
                            # Send partial transcript while still recording
                            if partial_text.strip():  # Only send if there's actual content
                                await websocket.send_text(json.dumps({
                                    "type": "transcript_partial",
                                    "data": partial_text
                                }))
                    except Exception as e:
                        print(f"Error during streaming transcription: {e}")
                        # We don't send error to client here to avoid interrupting the recording process

            # 2️ control / JSON frames
            elif message["type"] == "websocket.receive" and "text" in message:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_text(json.dumps({"type": "error", "data": "Malformed JSON"}))
                    continue

                if data.get("type") == "silence":
                    # ─ End‑of‑utterance: transcribe the accumulated audio ─
                    audio_buffer.seek(0) # Rewind buffer to the beginning
                    audio_bytes = audio_buffer.read()
                    
                    final_transcript = ""
                    if audio_bytes:
                        try:
                            # Convert bytes to float32 numpy array (assuming 16-bit PCM)
                            # Similar logic to load_audio_from_upload but without file reading
                            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                            
                            # Transcribe using faster-whisper model with streaming enabled
                            segments, info = model.transcribe(
                                audio_np, 
                                beam_size=5,
                                language="en",  # Can be made configurable
                                condition_on_previous_text=True
                            )
                            
                            print(f"Detected language '{info.language}' with probability {info.language_probability}")
                            
                            # Collect segments and send partial updates as they become available
                            segments_list = []
                            for segment in segments:
                                segments_list.append(segment)
                                
                                # Send partial transcript update
                                partial_transcript = "".join(seg.text for seg in segments_list)
                                await websocket.send_text(json.dumps({
                                    "type": "transcript_partial",
                                    "data": partial_transcript,
                                }))
                                
                                # Add a small delay to simulate streaming effect
                                await asyncio.sleep(0.05)
                            
                            final_transcript = "".join(segment.text for segment in segments_list)
                            
                        except Exception as e:
                            print(f"Error during faster-whisper transcription: {e}")
                            await websocket.send_text(json.dumps({"type": "error", "data": f"Transcription error: {e}"}))
                            final_transcript = "[Transcription Error]" # Indicate error in transcript

                    # Send the final transcript
                    await websocket.send_text(json.dumps({
                        "type": "transcript_final",
                        "data": final_transcript,
                    }))

                    # Reset the audio buffer for the next utterance
                    audio_buffer.seek(0)
                    audio_buffer.truncate()

                    # Kick off RAG/LLM if a command was pending
                    if pending_command_json is not None:
                        # Use a task to avoid blocking the websocket loop
                        asyncio.create_task(_handle_llm(websocket, final_transcript, pending_command_json))
                        pending_command_json = None # Clear pending command

                elif data.get("type") == "command":
                    # Store the command data, it will be used when 'silence' is received
                    pending_command_json = data.get("data") # Store the raw command JSON string
                    if not isinstance(pending_command_json, str):
                         print(f"Warning: Received command data is not a string: {pending_command_json}")
                         await websocket.send_text(json.dumps({"type": "error", "data": "Invalid command data format"}))
                         pending_command_json = None # Discard invalid command


            # ─ any other message type is ignored ─

    except WebSocketDisconnect:
        print("Client disconnected")
        manager.disconnect(websocket)
    except Exception as e: # Catch other potential errors
        print(f"WebSocket Error: {e}")
        manager.disconnect(websocket)
        # Attempt to send an error message if the socket is still open
        try:
            await websocket.send_text(json.dumps({"type": "error", "data": f"Server error: {e}"}))
        except:
            pass # Ignore if sending fails (socket likely closed)
    finally:
         # Ensure buffer is closed on exit
        audio_buffer.close()


async def _handle_llm(websocket: WebSocket, transcript: str, command_json_str: str):
    """Runs RAG + LLM, pushes token stream back to client."""
    try:
        # Ensure command_json_str is valid JSON before parsing
        try:
            # command_from_json expects a string containing JSON
            command: Command = command_from_json(command_json_str, question=transcript) 
        except json.JSONDecodeError as json_err:
             print(f"Error decoding command JSON: {json_err}")
             await websocket.send_text(json.dumps({"type": "error", "data": f"Invalid command JSON received: {json_err}"}))
             return
        except Exception as cmd_err: # Catch potential errors in command_from_json itself
             print(f"Error processing command: {cmd_err}")
             await websocket.send_text(json.dumps({"type": "error", "data": f"Error processing command: {cmd_err}"}))
             return

        if command is None:
             print("Command object is None after parsing.")
             await websocket.send_text(json.dumps({"type": "error", "data": "Failed to parse command object."}))
             return

        # Assemble prompt (assuming it returns a dict with 'response' key containing the final prompt string)
        prompt_data = assemble_prompt(command) 
        if not isinstance(prompt_data, dict) or "response" not in prompt_data:
             print(f"Error: assemble_prompt did not return expected format. Got: {prompt_data}")
             await websocket.send_text(json.dumps({"type": "error", "data": "Internal error assembling prompt."}))
             return
             
        final_prompt = prompt_data["response"]
        if not isinstance(final_prompt, str):
             print(f"Error: Assembled prompt is not a string. Got: {final_prompt}")
             await websocket.send_text(json.dumps({"type": "error", "data": "Internal error generating prompt text."}))
             return

        # Stream response from LLM
        async for tok in stream_chat_completion(final_prompt, model="openai"): # Assuming 'openai' is the intended model key
            await websocket.send_text(json.dumps({
                "type": "llm_token",
                "data": tok,
            }))
            
    except WebSocketDisconnect:
         print("Client disconnected during LLM streaming.")
         # Don't try to send error if disconnected
    except Exception as exc:  
        print(f"Error during LLM processing or streaming: {exc}")
        try:
            # Send error back to client if possible
            await websocket.send_text(json.dumps({"type": "error", "data": f"LLM processing error: {exc}"}))
        except WebSocketDisconnect:
             print("Client disconnected before LLM error could be sent.")
        except Exception as send_err:
             print(f"Failed to send LLM error to client: {send_err}")


async def stream_chat_completion(prompt: str, model: str = "openai") -> AsyncGenerator[str, None]:
    """
    Stream a chat completion from the LLM, yielding tokens as they become available.
    
    Args:
        prompt: The input prompt for the LLM
        model: The model provider to use ("openai" or "anthropic" etc.)
        
    Yields:
        Individual tokens from the LLM response as they become available
    """
    try:
        # Create LLM client using the configured model
        llm = create_llm(Config.get_instance().llm, stream=True)
        
        # Start token streaming
        print(f"Streaming response from LLM for prompt: {prompt[:50]}...")
        async for token in llm.astream(prompt):
            # If token is in a different format (e.g. dict), extract the actual token text
            if isinstance(token, dict):
                if "choices" in token and token["choices"] and "text" in token["choices"][0]:
                    token = token["choices"][0]["text"]
                elif "content" in token:
                    token = token["content"]
                    
            # Yield the token if it's a string
            if isinstance(token, str):
                yield token
                # Small delay to avoid flooding the WebSocket
                await asyncio.sleep(0.01)
            
    except Exception as e:
        print(f"Error streaming LLM response: {e}")
        yield f"[Error: {str(e)}]"