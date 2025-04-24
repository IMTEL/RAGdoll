import os
import logging
import uuid
import pytest
from src.context_upload import compute_embedding
from src.context_upload import process_file_and_store
from unittest.mock import patch



#UNCOMMENT THE FOLLOWING CODE AND REPLACE THE PLACEHOLDER VALUES WITH YOUR OWN VALUES TO UPLOAD FILE
    
# @pytest.mark.integration
# def test_process_file_and_store():
#     current_dir = os.path.dirname(__file__)  # Directory of the current test file.
#     file_path = os.path.join(current_dir, "test_sets", "bee_movie_script.txt")
#     category = "General Information"  # Updated to use category instead of NPC
#     success = process_file_and_store(file_path, category)
#     assert success == True
#     assert os.path.exists(file_path) == True
#     assert os.path.isfile(file_path) == True

#     assert success == True


# @pytest.fixture
# def mock_embedding():
#     return [0.1, 0.2, 0.3]  # Fake embedding

# @pytest.fixture
# def mock_post_context():
#     return True  # Simulate successful DB insert

# @patch("src.context_upload.get_database")
# @patch("src.context_upload.compute_embedding", return_value=[0.1, 0.2, 0.3])
# def test_process_file_and_store(mock_embedding, mock_db):
#     mock_db.return_value.post_context.return_value = True  # Mock DB success

#     file_path = "tests/test_sets/bee_movie_script.txt"
#     category = "General Information"  # Updated to use category instead of NPC

#     success = process_file_and_store(file_path, category)
    
#     assert success == True
    

# def test_retrieve_uploaded_data():
#     from src.rag_service.dao import get_database
    
#     db = get_database()  # Get the mock DB
#     data = db.data  # Access stored data
    
#     assert len(data) > 0  # Ensure at least one document was stored
#     assert data[0]["documentName"] == "bee_movie_script.txt"  # Check stored filename
#     assert data[0]["category"] == "General Information"  # Check the category
