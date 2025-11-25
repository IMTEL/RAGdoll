import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi_jwt_auth import AuthJWT

from src.config import Config
from src.context_upload import process_file_and_store
from src.globals import agent_dao, auth_service
from src.rag_service.dao.factory import get_document_dao
from src.utils.global_logs import progress_log


logger = logging.getLogger(__name__)

router = APIRouter()

config = Config()


def _process_document_background(
    file_location: Path,
    agent_id: str,
    embedding_model: str,
    embedding_api_key: str | None,
    file_size_bytes: int,
    filename: str,
    task_id: str,
):
    """Background task to process and store document.

    This function runs in a background thread, allowing the upload endpoint
    to return immediately while processing continues. Progress is tracked
    in the global progress_log.
    """
    from datetime import UTC, datetime

    try:
        # Update progress: started
        for entry in progress_log:
            if entry.get("task_id") == task_id:
                entry.update(
                    {
                        "status": "processing",
                        "started_at": datetime.now(UTC),
                        "message": f"Processing {filename}...",
                    }
                )
                break

        success, document_id = process_file_and_store(
            str(file_location),
            agent_id,
            embedding_model,
            file_size_bytes=file_size_bytes,
            embedding_api_key=embedding_api_key,
        )

        # Update progress: completed
        for entry in progress_log:
            if entry.get("task_id") == task_id:
                if success:
                    entry.update(
                        {
                            "status": "complete",
                            "completed_at": datetime.now(UTC),
                            "message": f"Successfully processed {filename}",
                            "document_id": document_id,
                        }
                    )
                    logger.info(
                        f"Background processing completed for '{filename}': "
                        f"document_id={document_id}, agent_id={agent_id}"
                    )
                else:
                    entry.update(
                        {
                            "status": "failed",
                            "completed_at": datetime.now(UTC),
                            "message": f"Failed to process {filename}",
                        }
                    )
                    logger.error(f"Background processing failed for '{filename}'")
                break

    except Exception as e:
        logger.error(f"Error in background processing of '{filename}': {e}")
        # Update progress: error
        for entry in progress_log:
            if entry.get("task_id") == task_id:
                entry.update(
                    {
                        "status": "error",
                        "completed_at": datetime.now(UTC),
                        "message": f"Error processing {filename}: {e!s}",
                    }
                )
                break
    finally:
        # Clean up temp file
        if file_location.exists():
            try:
                file_location.unlink()
                logger.debug(f"Cleaned up temp file: {file_location}")
            except Exception as e:
                logger.error(f"Failed to delete temp file {file_location}: {e}")


@router.post("/upload/agent")
async def upload_document_for_agent(
    agent_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),  # noqa: B008
    authorize: Annotated[AuthJWT, Depends()] = None,
):
    """Upload a document for a specific agent with role-based access control.

    This endpoint creates or updates a document within an agent's knowledge base.
    If a document with the same name already exists for this agent, it will be updated.
    Document filtering is controlled by agent roles' document_access lists.

    The document processing (chunking, embedding, storage) happens asynchronously
    in the background, so this endpoint returns immediately after uploading the file.

    Args:
        agent_id: Unique identifier of the agent
        background_tasks: FastAPI background tasks handler (injected)
        file: The uploaded document file (txt, md, pdf, docx, etc.)
        authorize (Annotated[AuthJWT, Depends()]): Jwt token object

    Returns:
        dict: Success message indicating upload received and processing started

    Raises:
        HTTPException: If authentication fails, agent not found, or file upload fails
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Verify agent exists and access key is valid
    auth_service.auth(authorize, agent_id)
    agent = agent_dao.get_agent_by_id(agent_id)

    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    # Create the directory if it does not exist
    temp_files_dir = Path("temp_files")
    temp_files_dir.mkdir(parents=True, exist_ok=True)

    # Save the file temporarily and capture its size
    file_location = temp_files_dir / Path(file.filename).name

    try:
        import uuid
        from datetime import UTC, datetime

        file_content = file.file.read()
        file_size_bytes = len(file_content)

        with open(file_location, "wb") as buffer:
            buffer.write(file_content)

        # Create a task ID for progress tracking
        task_id = str(uuid.uuid4())

        # Initialize progress entry
        progress_log.append(
            {
                "task_id": task_id,
                "task_name": f"upload_{file.filename}",
                "filename": file.filename,
                "agent_id": agent_id,
                "status": "queued",
                "started_at": None,
                "completed_at": None,
                "message": f"Queued for processing: {file.filename}",
                "document_id": None,
                "created_at": datetime.now(UTC),
            }
        )

        # Schedule background processing
        background_tasks.add_task(
            _process_document_background,
            file_location=file_location,
            agent_id=agent_id,
            embedding_model=agent.embedding_model,
            embedding_api_key=agent.embedding_api_key,
            file_size_bytes=file_size_bytes,
            filename=file.filename,
            task_id=task_id,
        )

        logger.info(
            f"File '{file.filename}' uploaded for agent '{agent_id}', "
            f"processing started in background (task_id: {task_id})"
        )

        return {
            "message": "Document uploaded successfully, processing in background",
            "filename": file.filename,
            "agent_id": agent_id,
            "task_id": task_id,
            "status": "queued",
            "size_bytes": file_size_bytes,
        }

    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        # Clean up temp file if upload failed
        if file_location.exists():
            file_location.unlink()
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/upload/status/{task_id}")
async def get_upload_status(task_id: str):
    """Get the status of a background document upload/processing task.

    Args:
        task_id: The task ID returned from the upload endpoint

    Returns:
        dict: Status information about the upload task

    Raises:
        HTTPException: If task not found
    """
    for entry in progress_log:
        if entry.get("task_id") == task_id:
            return {
                "task_id": task_id,
                "filename": entry.get("filename"),
                "agent_id": entry.get("agent_id"),
                "status": entry.get("status"),
                "message": entry.get("message"),
                "document_id": entry.get("document_id"),
                "started_at": entry.get("started_at"),
                "completed_at": entry.get("completed_at"),
            }

    raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")


@router.get("/documents/agent")
async def get_documents_for_agent(
    agent_id: str, authorize: Annotated[AuthJWT, Depends()] = None
):
    """Retrieve all documents associated with a specific agent.

    This endpoint returns metadata for all documents that belong to the agent,
    including document IDs, names, and timestamps.

    Args:
        agent_id: Unique identifier of the agent
        authorize (Annotated[AuthJWT, Depends()]): Jwt token object

    Returns:
        dict: List of documents with their metadata

    Raises:
        HTTPException: If agent not found or retrieval fails
    """
    auth_service.auth(authorize, agent_id)

    # Verify agent exists
    agent = agent_dao.get_agent_by_id(agent_id)

    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    try:
        # Fetch all documents for this agent
        document_dao = get_document_dao()
        documents = document_dao.get_by_agent_id(agent_id)

        # Convert to response format
        document_list = []
        for doc in documents:
            try:
                # Format size in a human-readable way
                size_bytes = getattr(doc, "size_bytes", 0)
                if size_bytes < 1024:
                    size_str = f"{size_bytes} B"
                elif size_bytes < 1024 * 1024:
                    size_str = f"{size_bytes / 1024:.2f} KB"
                elif size_bytes < 1024 * 1024 * 1024:
                    size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
                else:
                    size_str = f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

                document_list.append(
                    {
                        "id": doc.id,
                        "name": doc.name,
                        "size": size_str,
                        "size_bytes": size_bytes,
                        "created_at": doc.created_at.isoformat()
                        if doc.created_at
                        else None,
                        "updated_at": doc.updated_at.isoformat()
                        if doc.updated_at
                        else None,
                    }
                )

            except AttributeError as attr_err:
                logger.error(f"Document missing required field: {attr_err}, doc: {doc}")
                # Skip this document and continue with others
                continue

        logger.info(f"Retrieved {len(document_list)} documents for agent '{agent_id}'")

        return {
            "agent_id": agent_id,
            "document_count": len(document_list),
            "documents": document_list,
        }

    except Exception as e:
        logger.error(f"Error retrieving documents for agent '{agent_id}': {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/documents/")
async def delete_document(
    document_id: str, agent_id: str, authorize: Annotated[AuthJWT, Depends()] = None
):
    """Delete a document and all its associated context chunks.

    This endpoint permanently removes a document and cascades the deletion
    to all context chunks associated with it. It also removes the document ID
    from all agent roles' document_access lists.

    Args:
        document_id: Unique identifier of the document to delete
        agent_id: id of agent
        authorize (Annotated[AuthJWT, Depends()]): Jwt token object

    Returns:
        dict: Success message with deletion details

    Raises:
        HTTPException: If document not found or deletion fails
    """
    auth_service.auth(authorize, agent_id)
    try:
        document_dao = get_document_dao()

        # Check if document exists first
        document = document_dao.get_by_id(document_id)
        if not document:
            raise HTTPException(
                status_code=404, detail=f"Document '{document_id}' not found"
            )

        # Get the agent that owns this document
        agent = agent_dao.get_agent_by_id(document.agent_id)
        if agent.id != agent_id:
            raise HTTPException(status_code=401, detail="Missmatch in agent_ids")

        if agent:
            updated = False
            for role in agent.roles:
                if document_id in role.document_access:
                    role.document_access.remove(document_id)
                    updated = True

            if updated:
                agent_dao.add_agent(agent)  # This updates the existing agent
                logger.info(
                    f"Removed document '{document_id}' from agent '{agent.id}' roles"
                )
            else:
                logger.info(
                    f"Document '{document_id}' not found in any roles of agent '{agent.id}'"
                )

        # Delete the document (this also deletes associated contexts)
        success = document_dao.delete(document_id)

        if success:
            logger.info(f"Successfully deleted document '{document_id}'")
            return {
                "message": "Document deleted successfully",
                "document_id": document_id,
                "document_name": document.name,
                "removed_from_roles": updated if agent else False,
            }
        else:
            raise HTTPException(
                status_code=500, detail=f"Failed to delete document '{document_id}'"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document '{document_id}': {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
