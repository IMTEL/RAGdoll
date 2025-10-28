from unittest.mock import MagicMock, mock_open, patch

from src.context_upload import compute_embedding, process_file_and_store


@patch("src.context_upload.create_embeddings_model")
def test_compute_embedding_calls_model(mock_create_model):
    mock_model = MagicMock()
    mock_model.get_embedding.return_value = [0.1, 0.2, 0.3]
    mock_create_model.return_value = mock_model

    result = compute_embedding("hello world", "gemini:models/text-embedding-004")
    assert result is not None
    assert result == [0.1, 0.2, 0.3]
    mock_create_model.assert_called_once_with("gemini:models/text-embedding-004")
    mock_model.get_embedding.assert_called_once_with("hello world")


@patch("os.path.exists", return_value=False)
def test_process_file_not_found(mock_exists):
    result, doc_id = process_file_and_store(
        "missing.txt", "test-agent", "gemini:models/text-embedding-004"
    )
    assert result is False


@patch("os.path.exists", return_value=True)
def test_process_file_unsupported_type(mock_exists):
    result, doc_id = process_file_and_store(
        "file.pdf", "test-agent", "gemini:models/text-embedding-004"
    )
    assert result is False


@patch("src.rag_service.dao.factory.get_document_dao")
@patch("src.rag_service.dao.factory.get_context_dao")
@patch("src.context_upload.compute_embedding", side_effect=Exception("Embedding error"))
@patch("builtins.open", new_callable=mock_open, read_data="text")
@patch("os.path.exists", return_value=True)
def test_process_file_embedding_failure(
    mock_exists, mock_open_func, mock_embed, mock_context_dao, mock_doc_dao
):
    result, doc_id = process_file_and_store(
        "file.txt", "test-agent", "gemini:models/text-embedding-004"
    )
    assert result is False
