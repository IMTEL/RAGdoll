import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi_jwt_auth import AuthJWT

from src.config import Config
from src.context_upload import process_file_and_store
from src.globals import agent_dao, auth_service
from src.models.errors import EmbeddingAPIError, EmbeddingError
from src.rag_service.dao.factory import get_document_dao


logger = logging.getLogger(__name__)

router = APIRouter()

config = Config()


@router.post("/upload/agent")
async def upload_document_for_agent(
    agent_id: str,
    file: UploadFile = File(...),  # noqa: B008
    authorize: Annotated[AuthJWT, Depends()] = None,
):
    """Upload a document for a specific agent with role-based access control.

    This endpoint creates or updates a document within an agent's knowledge base.
    If a document with the same name already exists for this agent, it will be updated.
    Document filtering is controlled by agent roles' document_access lists.

    Args:
        agent_id: Unique identifier of the agent
        file: The uploaded document file (txt, md)
        authorize (Annotated[AuthJWT, Depends()]): Jwt token object

    Returns:
        dict: Success message with document details

    Raises:
        HTTPException: If authentication fails, agent not found, or processing fails
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
    file_content = file.file.read()
    file_size_bytes = len(file_content)

    with open(file_location, "wb") as buffer:
        buffer.write(file_content)

    try:
        # Process and store/update document with file size using agent's embedding model
        success, document_id = process_file_and_store(
            str(file_location),
            agent_id,
            agent.embedding_model,
            file_size_bytes=file_size_bytes,
        )

        # Delete temporary file
        file_location.unlink()

        logger.info(f"Uploaded file: {file.filename} for agent: {agent_id}")
        logger.info(f"Uploaded file: {file.filename} for agent: {agent_id}")

        if success:
            return {
                "message": "Document uploaded and processed successfully",
                "filename": file.filename,
                "agent_id": agent_id,
                "document_id": document_id,
                "status": "stored_with_embeddings",
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to process document")

    except EmbeddingAPIError as e:
        logger.error(f"Embedding API authentication error: {e}")
        raise HTTPException(
            status_code=401,
            detail=f"Embedding API authentication failed: {e!s}. "
            f"Please verify the API key for {e.provider} has access to model '{e.model}'.",
        ) from e
    except EmbeddingError as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Embedding error: {e!s}",
        ) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        # Clean up temp file if it exists
        if file_location.exists():
            file_location.unlink()


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
