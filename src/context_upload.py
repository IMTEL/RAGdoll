import logging
import os
import threading
import uuid
from collections.abc import Callable

from pymongo.errors import PyMongoError

from src.config import Config
from src.models.errors import EmbeddingAPIError, EmbeddingError
from src.rag_service.context import Context
from src.rag_service.dao.context.faiss_vector_store import FAISSIndexError
from src.rag_service.dao.factory import get_context_dao, get_document_dao
from src.rag_service.embeddings import create_embeddings_model
from src.scraper_service.scraper import ScraperService


logger = logging.getLogger(__name__)


# Load configuration and initialize services
config = Config()
# Increased chunk_size to 1000 and overlap to 100 for better context and fewer mid-sentence cuts
scraper_service = ScraperService(chunk_size=4000, overlap=200)


# One lock per (agent_id, document_name) so two uploads of the same document
# cannot interleave resolve -> delete old chunks -> insert new chunks.
# In-process only: with several worker processes this does not serialize across
# them. Entries are never evicted (one small Lock per distinct document name).
_document_locks: dict[tuple[str, str], threading.Lock] = {}
_document_locks_guard = threading.Lock()


def _document_lock(agent_id: str, document_name: str) -> threading.Lock:
    """Return the lock serializing processing of one agent's named document."""
    with _document_locks_guard:
        return _document_locks.setdefault((agent_id, document_name), threading.Lock())


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
    embeddings = embedding_model.get_embedding(text, task="document")
    return embeddings


def process_file_and_store(
    file_path: str,
    agent_id: str,
    embedding_model: str,
    document_id: str | None = None,
    file_size_bytes: int | None = None,
    embedding_api_key: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    document_name: str | None = None,
    kind: str | None = None,
) -> tuple[bool, str]:
    """Processes various file types, extracts and chunks text, computes embeddings, and stores data in the database.

    Uses the ScraperService to handle multiple file formats including:
    - PDF (.pdf)
    - Word documents (.docx, .doc)
    - PowerPoint presentations (.pptx, .ppt)
    - Excel spreadsheets (.xlsx, .xls)
    - Text files (.txt)
    - Markdown files (.md)
    - HTML files (.html, .htm)
    - And more

    The document is automatically chunked for optimal RAG performance.
    If document_id is provided and a document with that ID exists, it updates the document.
    Otherwise, creates a new document.

    Args:
        file_path (str): Path to the file to process.
        agent_id (str): Agent ID that owns this document.
        embedding_model (str): Embedding model to use in format "provider:model_name".
        document_id (str | None): Optional document ID for updates.
        file_size_bytes (int | None): Size of the file in bytes. If None, will be computed from file_path.
        embedding_api_key (str | None, optional): API key for the embedding model. Defaults to None.
        progress_callback: Optional callback receiving (percent, message).
        document_name (str | None): Name to store the document under. Defaults to
            the basename of file_path; uploads pass the client's filename because
            the file on disk has a unique, unrelated name.
        kind (str | None): "world_lore" or "educational". None keeps the kind of
            an existing document, or makes a new one educational. Every stored
            chunk carries the document's kind so prompts can label it.

    Returns:
        tuple[bool, str]: (Success status, Document ID)
    """
    name = document_name or os.path.basename(file_path)
    with _document_lock(agent_id, name):
        return _process_file_and_store_locked(
            file_path,
            agent_id,
            embedding_model,
            name,
            document_id,
            file_size_bytes,
            embedding_api_key,
            progress_callback,
            kind,
        )


def _process_file_and_store_locked(
    file_path: str,
    agent_id: str,
    embedding_model: str,
    document_name: str,
    document_id: str | None,
    file_size_bytes: int | None,
    embedding_api_key: str | None,
    progress_callback: Callable[[int, str], None] | None,
    kind: str | None = None,
) -> tuple[bool, str]:
    from src.models.rag import Document
    from src.models.rag.models import DEFAULT_DOCUMENT_KIND

    logger.info(f"Starting file processing for: {file_path} with agent_id: {agent_id}")

    # Verify file exists
    if not os.path.exists(file_path):
        logger.error(f"File '{file_path}' does not exist.")
        return False, ""

    # Check if file type is supported by scraper
    if not scraper_service.is_supported_file(file_path):
        logger.error(f"Unsupported file type: {os.path.splitext(file_path)[1]}")
        return False, ""

    # Compute file size if not provided
    if file_size_bytes is None:
        file_size_bytes = os.path.getsize(file_path)

    # Get database instances
    document_dao = get_document_dao()
    context_dao = get_context_dao()

    try:
        # Use scraper service to extract and chunk the document
        logger.info(f"Scraping file with ScraperService: {file_path}")
        if progress_callback:
            progress_callback(15, f"Extracting text from {document_name}...")
        scraped_documents = scraper_service.scrape_file(file_path)

        if not scraped_documents:
            logger.error(f"No content extracted from file: {file_path}")
            return False, ""

        logger.info(f"Extracted {len(scraped_documents)} chunks from {file_path}")

        # Check if we're updating an existing document
        existing_doc: Document | None = None
        if document_id:
            existing_doc = document_dao.get_by_id(document_id)
        else:
            # Check by name
            existing_doc = document_dao.get_by_name_and_agent(document_name, agent_id)

        replacing = existing_doc is not None
        if existing_doc is not None:
            # Update existing document. The new chunks are stored first and the
            # old ones removed afterwards, so a failed upload keeps the old version.
            document_id = existing_doc.id
            assert document_id is not None  # Type narrowing for mypy
            logger.info(f"Updating existing document '{document_id}'")
            document_kind = kind or existing_doc.kind
        else:
            # Create new document
            document_id = str(uuid.uuid4())
            logger.info(f"Creating new document '{document_id}'")
            document_kind = kind or DEFAULT_DOCUMENT_KIND

            doc = Document(
                id=document_id,
                name=document_name,
                agent_id=agent_id,
                size_bytes=file_size_bytes,
                kind=document_kind,
            )
            document_dao.create(doc)

        # Process and store each chunk
        total_chunks = len(scraped_documents)
        new_chunk_ids: list[str] = []
        try:
            for chunk_idx, scraped_doc in enumerate(scraped_documents):
                try:
                    # Update progress: 20% + (chunk_idx / total_chunks) * 75%
                    progress_percent = 20 + int((chunk_idx / total_chunks) * 75)
                    if progress_callback:
                        progress_callback(
                            progress_percent,
                            f"Processing chunk {chunk_idx + 1}/{total_chunks}...",
                        )

                    # Compute embedding for this chunk using the agent's embedding model
                    try:
                        embedding = compute_embedding(
                            scraped_doc.content, embedding_model, embedding_api_key
                        )
                    except (EmbeddingAPIError, EmbeddingError):
                        # Re-raise embedding-specific errors so they can be handled properly
                        raise
                    except ValueError as e:
                        # Invalid embedding model format
                        logger.error(
                            f"Invalid embedding model format for chunk {chunk_idx}: {e}"
                        )
                        raise EmbeddingError(
                            f"Invalid embedding model configuration: {e!s}", e
                        ) from e

                    # Store context for this chunk. Globally unique id: independent
                    # of the (temp) file path so identically named uploads and
                    # re-uploads never collide. Recorded before inserting so a
                    # half-stored chunk is rolled back too.
                    chunk_id = f"{document_id}:{chunk_idx}:{uuid.uuid4().hex[:12]}"
                    new_chunk_ids.append(chunk_id)
                    context_dao.insert_context(
                        document_id=document_id,
                        agent_id=agent_id,
                        embedding=embedding,
                        context=Context(
                            text=scraped_doc.content,
                            document_name=document_name,
                            document_id=document_id,
                            chunk_id=chunk_id,
                            chunk_index=chunk_idx,
                            total_chunks=total_chunks,
                            kind=document_kind,
                        ),
                    )
                    logger.debug(
                        f"Stored chunk {chunk_idx + 1}/{total_chunks} for document '{document_id}'"
                    )

                except (EmbeddingAPIError, EmbeddingError, FAISSIndexError):
                    # Re-raise embedding and vector index errors: the upload failed
                    raise
                except Exception as e:
                    logger.error(
                        f"Error processing chunk {chunk_idx} of '{file_path}': {e}"
                    )
                    if replacing:
                        # A partial new version must not replace the old one.
                        raise
                    # Continue processing other chunks
                    continue
        except Exception:
            if replacing and new_chunk_ids:
                # Leave the old version as it was: drop what was inserted.
                try:
                    context_dao.delete_chunks(agent_id, new_chunk_ids)
                except (PyMongoError, FAISSIndexError) as cleanup_error:
                    logger.error(
                        f"Could not remove partial new chunks of '{document_id}': "
                        f"{cleanup_error}"
                    )
            raise

        if replacing:
            deleted = context_dao.delete_by_document_id(
                document_id, agent_id=agent_id, keep_chunk_ids=set(new_chunk_ids)
            )
            logger.info(f"Deleted {deleted} old contexts of document '{document_id}'")
            if existing_doc is not None and existing_doc.kind != document_kind:
                # Only after the new chunks are in, so a failed upload keeps
                # the old kind together with the old chunks.
                existing_doc.kind = document_kind
                document_dao.update(existing_doc)

        logger.info(
            f"Successfully stored '{document_name}' with {total_chunks} chunks into the database."
        )

        if progress_callback:
            progress_callback(100, f"Successfully processed {document_name}")

    except (EmbeddingAPIError, EmbeddingError):
        # Re-raise embedding errors
        raise
    except Exception as e:
        logger.error(f"Error processing file '{file_path}': {e}")
        return False, ""

    return True, document_id


if __name__ == "__main__":
    # Example usage of the process_file_and_store function.
    file_path = "src.context_files.salmon.txt"
    agent_id = "test-agent-123"
    embedding_model = "gemini:models/text-embedding-004"
    success, doc_id = process_file_and_store(file_path, agent_id, embedding_model)
    if success:
        print(f"Document stored with ID: {doc_id}")
