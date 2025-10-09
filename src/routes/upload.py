import logging
import os
from enum import Enum
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.context_upload import process_file_and_store


# Configure logging
logging.basicConfig(level=logging.INFO)

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
    file_location = ""
    try:
        # Create the directory if it does not exist
        temp_files_dir = Path("temp_files")
        temp_files_dir.mkdir(parents=True, exist_ok=True)

        if not file.filename:
            raise HTTPException(status_code=400, detail="No file uploaded")

        # Save the file temporarily
        # TODO: Why temporarily save to disk?
        file_location = temp_files_dir / Path(file.filename).name
        with open(file_location, "wb") as buffer:
            buffer.write(file.file.read())

        # Process and store - convert enum to its string value for database compatibility
        success = process_file_and_store(str(file_location), category.value)

        # Delete temporary file
        file_location.unlink()

        logging.info(f"Uploaded file: {file.filename}, Category: {category.value}")

        if success:
            return {
                "message": "File uploaded and processed successfully",
                "filename": file.filename,
                "category": category.value,
                "status": "stored_with_embeddings",
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to process file")

    except Exception as e:
        # Clean up temp file if it exists
        if file_location and os.path.exists(file_location):
            os.remove(file_location)
        raise HTTPException(status_code=500, detail=str(e)) from e
