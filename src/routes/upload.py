import logging
from enum import Enum
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.context_upload import process_file_and_store
from src.rag_service.dao.factory import get_agent_dao


logger = logging.getLogger(__name__)

router = APIRouter()


class DocumentCategory(Enum):
    # Other categories
    BEHAVIORAL_CONTEXT = "Behavioral Context"
    GENERAL_INFORMATION = "General Information"
    CONVERSATIONAL_ASSETS = "Conversational Assets"
    TRAINING_DATA = "Training Data"
    MISCELLANEOUS = "Miscellaneous"

    # Scene categories from Blue Sector
    FISH_FEEDING = "FishFeeding"
    LABORATORY = "Laboratory"
    FISH_FACTORY = "FishFactory"
    FISH_WELFARE = "FishWelfare"
    FISH_MAINTENANCE = "FishMaintenance"
    OCEAN = "Ocean"


@router.post("/upload/")
async def upload_document(
    file: UploadFile = File(...),  # noqa: B008
    category: DocumentCategory = DocumentCategory.GENERAL_INFORMATION,
):
    """API to receive a document, process it, and store it with RAG embeddings.

    **DEPRECATED**: Use POST /upload/agent/{agent_id} instead.
    This endpoint will be removed in a future version.

    This endpoint:
    1. Accepts text or markdown files
    2. Computes embeddings for semantic search
    3. Stores the document with associated category/corpus ID
    4. Enables later retrieval through RAG queries

    Args:
        file: The uploaded document file (txt, md)
        category: The document category/corpus identifier from DocumentCategory enum

    Returns:
        dict: Success message with category and filename information

    Raises:
        HTTPException: If file processing fails or unsupported file type
    """
    logger.warning(
        "Using deprecated endpoint /upload/ - please migrate to /upload/agent/{agent_id}"
    )

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Create the directory if it does not exist
    temp_files_dir = Path("temp_files")
    temp_files_dir.mkdir(parents=True, exist_ok=True)

    # Save the file temporarily
    # TODO: Why temporarily save to disk?
    file_location = temp_files_dir / Path(file.filename).name
    with open(file_location, "wb") as buffer:
        buffer.write(file.file.read())

    try:
        # For backward compatibility, use a default agent
        # In production, this should fail or require agent_id
        default_agent_id = "legacy-agent"
        categories = [category.value]

        # Process and store
        success, document_id = process_file_and_store(
            str(file_location), default_agent_id, categories
        )

        # Delete temporary file
        file_location.unlink()

        logger.info(f"Uploaded file: {file.filename}, Category: {category.value}")

        if success:
            return {
                "message": "File uploaded and processed successfully (using deprecated endpoint)",
                "filename": file.filename,
                "category": category.value,
                "document_id": document_id,
                "status": "stored_with_embeddings",
                "warning": "This endpoint is deprecated. Use POST /upload/agent/{agent_id} instead.",
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to process file")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        # Clean up temp file if it exists
        if file_location.exists():
            file_location.unlink()


@router.post("/upload/agent/{agent_id}")
async def upload_document_for_agent(
    agent_id: str,
    file: UploadFile = File(...),  # noqa: B008
    access_key: str = Form(...),
    categories: str = Form(...),  # Comma-separated list
):
    """Upload a document for a specific agent with role-based access control.

    This endpoint creates or updates a document within an agent's knowledge base.
    If a document with the same name already exists for this agent, it will be updated.

    Args:
        agent_id: Unique identifier of the agent
        file: The uploaded document file (txt, md)
        access_key: API key authorized to modify this agent
        categories: Comma-separated list of category tags (e.g., "General,Technical")

    Returns:
        dict: Success message with document details

    Raises:
        HTTPException: If authentication fails, agent not found, or processing fails
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Verify agent exists and access key is valid
    agent_dao = get_agent_dao()
    agent = agent_dao.get_agent_by_id(agent_id)

    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    if not agent.is_access_key_valid(access_key):
        raise HTTPException(status_code=403, detail="Invalid access key")

    # Parse categories
    category_list = [cat.strip() for cat in categories.split(",") if cat.strip()]
    if not category_list:
        raise HTTPException(status_code=400, detail="At least one category is required")

    # Create the directory if it does not exist
    temp_files_dir = Path("temp_files")
    temp_files_dir.mkdir(parents=True, exist_ok=True)

    # Save the file temporarily
    file_location = temp_files_dir / Path(file.filename).name
    with open(file_location, "wb") as buffer:
        buffer.write(file.file.read())

    try:
        # Process and store/update document
        success, document_id = process_file_and_store(
            str(file_location), agent_id, category_list
        )

        # Delete temporary file
        file_location.unlink()

        logger.info(
            f"Uploaded file: {file.filename} for agent: {agent_id}, Categories: {category_list}"
        )

        if success:
            return {
                "message": "Document uploaded and processed successfully",
                "filename": file.filename,
                "agent_id": agent_id,
                "document_id": document_id,
                "categories": category_list,
                "status": "stored_with_embeddings",
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to process document")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        # Clean up temp file if it exists
        if file_location.exists():
            file_location.unlink()
