import logging
import os
import uuid

from src.config import Config
from src.models.errors import EmbeddingAPIError, EmbeddingError
from src.rag_service.context import Context
from src.rag_service.dao.factory import get_context_dao, get_document_dao
from src.rag_service.embeddings import create_embeddings_model


logger = logging.getLogger(__name__)


# Load configuration
config = Config()


def compute_embedding(
    text: str, embedding_model_str: str, embedding_api_key: str | None = None
) -> list[float]:
    """Computes an embedding for the given text using the specified embedding model and API key.

    Args:
        text (str): The text to embed.
        embedding_model_str (str): The embedding model to use in format "provider:model_name".
        embedding_api_key (str | None): The embedding API key to use.

    Returns:
        list[float]: The computed embedding as a list of floats.
    """
    embedding_model = create_embeddings_model(
        embedding_model_str, embedding_api_key=embedding_api_key
    )
    embeddings = embedding_model.get_embedding(text)
    return embeddings


def process_file_and_store(
    file_path: str,
    agent_id: str,
    embedding_model: str,
    document_id: str | None = None,
    file_size_bytes: int | None = None,
    embedding_api_key: str | None = None,
) -> tuple[bool, str]:
    """Processes a .txt or .md file, extracts its text, computes its embedding, and stores the data in the database.

    Currently stores the whole document as a single context entry.
    If document_id is provided and a document with that ID exists, it updates the document.
    Otherwise, creates a new document.

    TODO: Implement text scraping and chunking for large documents
    Future enhancements should include:
    - Intelligent text chunking based on semantic boundaries (paragraphs, sections)
    - Chunk size optimization (e.g., 512-1024 tokens per chunk)
    - Overlap between chunks for context preservation
    - Store multiple chunk entries per document with proper metadata
    - Update retrieval logic to handle chunk reassembly

    Args:
        file_path (str): Path to the text or markdown file.
        agent_id (str): Agent ID that owns this document.
        embedding_model (str): Embedding model to use in format "provider:model_name".
        document_id (str | None): Optional document ID for updates.
        file_size_bytes (int | None): Size of the file in bytes. If None, will be computed from file_path.
        embedding_api_key (str | None, optional): API key for the embedding model. Defaults to None.

    Returns:
        tuple[bool, str]: (Success status, Document ID)
    """
    from src.models.rag import Document

    logger.info(f"Starting file processing for: {file_path} with agent_id: {agent_id}")

    # Verify file exists
    if not os.path.exists(file_path):
        logger.error(f"File '{file_path}' does not exist.")
        return False, ""

    # Verify file extension is supported.
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in [".txt", ".md"]:
        logger.error("Unsupported file type. Only .txt and .md files are supported.")
        return False, ""

    # Extract the file's text content.
    # TODO: Handle different encodings more gracefully.
    try:
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, encoding="latin-1") as f:
                text = f.read()
        except Exception as e:
            logger.error(f"Error reading file '{file_path}': {e}")
            return False, ""
    except Exception as e:
        logger.error(f"Error reading file '{file_path}': {e}")
        return False, ""

    # Compute the embedding using the agent's embedding model.
    # TODO: For chunked documents, compute embeddings per chunk
    try:
        embedding = compute_embedding(
            text, embedding_model, embedding_api_key=embedding_api_key
        )
    except (EmbeddingAPIError, EmbeddingError):
        # Re-raise embedding-specific errors so they can be handled properly
        raise
    except ValueError as e:
        # Invalid embedding model format
        logger.error(f"Invalid embedding model format for file '{file_path}': {e}")
        raise EmbeddingError(f"Invalid embedding model configuration: {e!s}", e) from e
    except Exception as e:
        logger.error(f"Error computing embedding for file '{file_path}': {e}")
        return False, ""

    # Use the file's basename as the document name.
    document_name = os.path.basename(file_path)

    # Compute file size if not provided
    if file_size_bytes is None:
        file_size_bytes = os.path.getsize(file_path)

    # Get database instances
    document_dao = get_document_dao()
    context_dao = get_context_dao()

    try:
        # Check if we're updating an existing document
        existing_doc: Document | None = None
        if document_id:
            existing_doc = document_dao.get_by_id(document_id)
        else:
            # Check by name
            existing_doc = document_dao.get_by_name_and_agent(document_name, agent_id)

        if existing_doc is not None:
            # Update existing document
            document_id = existing_doc.id
            assert document_id is not None  # Type narrowing for mypy
            logger.info(f"Updating existing document '{document_id}'")

            # Delete old contexts for this document
            if hasattr(context_dao, "collection"):
                # TODO: make this generic by adding a delete_by_document_id method to ContextDAO
                # For MongoDB implementation
                context_collection = context_dao.collection  # type: ignore[attr-defined]
                if hasattr(context_collection, "delete_many"):
                    result = context_collection.delete_many(
                        {"document_id": document_id}
                    )
                    logger.info(f"Deleted {result.deleted_count} old contexts")
        else:
            # Create new document
            document_id = str(uuid.uuid4())
            logger.info(f"Creating new document '{document_id}'")

            doc = Document(
                id=document_id,
                name=document_name,
                agent_id=agent_id,
                size_bytes=file_size_bytes,
            )
            document_dao.create(doc)

        # Store context for the document
        # TODO: When implementing chunking, loop through chunks and store each with proper metadata
        context_dao.insert_context(
            document_id=document_id,
            agent_id=agent_id,
            embedding=embedding,
            context=Context(
                text=text,
                document_name=document_name,
                document_id=document_id,
                chunk_id=None,  # Not chunked yet
                chunk_index=0,  # First (and only) chunk
                total_chunks=1,  # Only one chunk (whole document)
            ),
        )
    except Exception as e:
        logger.error(f"Error inserting context into database: {e}")
        return False, ""

    logger.info(f"Successfully stored '{document_name}' into the database.")

    return True, document_id


if __name__ == "__main__":
    # Example usage of the process_file_and_store function.
    file_path = "src.context_files.salmon.txt"
    agent_id = "test-agent-123"
    embedding_model = "gemini:models/text-embedding-004"
    success, doc_id = process_file_and_store(file_path, agent_id, embedding_model)
    if success:
        print(f"Document stored with ID: {doc_id}")
