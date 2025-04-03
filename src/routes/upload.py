import os
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException
from src.context_upload import process_file_and_store

# Configure logging
logging.basicConfig(level=logging.INFO)

router = APIRouter()

@router.post("/upload/")
async def upload_document(file: UploadFile = File(...), NPC: int = 0):
    """
    API to receive a document, process it, and store it.
    """
    try:
        # Create the directory if it does not exist
        os.makedirs("temp_files", exist_ok=True)

        # Save the file temporarily
        file_location = f"temp_files/{file.filename}"
        with open(file_location, "wb") as buffer:
            buffer.write(file.file.read())

        # Process and store
        success = process_file_and_store(file_location, NPC)

        # Delete temporary file
        os.remove(file_location)

        if success:
            return {"message": "File uploaded and processed successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to process file")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))