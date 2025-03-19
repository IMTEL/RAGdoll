
import os
import logging
import uuid
import pytest
from src.context_upload import compute_embedding
from src.context_upload import process_file_and_store


# UNCOMMENT THE FOLLOWING CODE AND REPLACE THE PLACEHOLDER VALUES WITH YOUR OWN VALUES TO UPLOAD FILE
    
# @pytest.mark.integration
# def test_process_file_and_store():
#     current_dir = os.path.dirname(__file__)  # Directory of the current test file.
#     file_path = os.path.join(current_dir, "test_sets", "salmon.txt")
#     NPC = 100
#     success = process_file_and_store(file_path, NPC)
#     assert success == True
#     assert os.path.exists(file_path) == True
#     assert os.path.isfile(file_path) == True
  
#     assert success == True
   
    
    