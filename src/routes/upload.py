import os
import logging
from enum import Enum
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException
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
    file: UploadFile = File(...), 
    category: Optional[DocumentCategory] = DocumentCategory.GENERAL_INFORMATION
):
    """
    API to receive a document, process it, and store it.
    
    Args:
        file: The uploaded document file (txt, md)
        category: The document category from the DocumentCategory enum
    """
    try:
        # Create the directory if it does not exist
        os.makedirs("temp_files", exist_ok=True)

        # Save the file temporarily
        file_location = f"temp_files/{file.filename}"
        with open(file_location, "wb") as buffer:
            buffer.write(file.file.read())

        # Process and store - convert enum to its string value for database compatibility
        success = process_file_and_store(file_location, category_value)

        # Delete temporary file
        os.remove(file_location)

        if success:
            return {"message": "File uploaded and processed successfully", "category": category_value}
        else:
            raise HTTPException(status_code=500, detail="Failed to process file")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))