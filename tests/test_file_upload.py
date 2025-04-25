import pytest
from unittest.mock import patch, MagicMock, mock_open
from src.context_upload import compute_embedding, process_file_and_store


@patch("src.context_upload.embedding_model")
def test_compute_embedding_calls_model(mock_model):
    mock_model.get_embedding.return_value = [0.1, 0.2, 0.3]
    result = compute_embedding("hello world")
    assert result != None
    assert result == [0.1, 0.2, 0.3]
    mock_model.get_embedding.assert_called_once_with("hello world")


@patch("os.path.exists", return_value=False)
def test_process_file_not_found(mock_exists):
    result = process_file_and_store("missing.txt", category="General Information")
    assert result is False

@patch("os.path.exists", return_value=True)
def test_process_file_unsupported_type(mock_exists):
    result = process_file_and_store("file.pdf", category="General Information")
    assert result is False

@patch("src.rag_service.dao.get_database")
@patch("src.context_upload.compute_embedding", side_effect=Exception("Embedding error"))
@patch("builtins.open", new_callable=mock_open, read_data="text")
@patch("os.path.exists", return_value=True)
def test_process_file_embedding_failure(mock_exists, mock_open_func, mock_embed, mock_get_db):
    result = process_file_and_store("file.txt", category="General Information")
    assert result is False

