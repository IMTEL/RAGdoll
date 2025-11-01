import logging
import os
import uuid

from src.config import Config
from src.models.errors import EmbeddingAPIError, EmbeddingError
from src.rag_service.context import Context
from src.rag_service.dao.factory import get_context_dao, get_document_dao
from src.rag_service.embeddings import create_embeddings_model
from src.scraper_service.scraper import ScraperService


logger = logging.getLogger(__name__)


# Load configuration and initialize services
config = Config()
# Increased chunk_size to 1000 and overlap to 100 for better context and fewer mid-sentence cuts
scraper_service = ScraperService(chunk_size=1000, overlap=100)


def compute_embedding(text: str, embedding_model_str: str) -> list[float]:
    """Computes an embedding for the given text using the specified embedding model.

    Args:
        text (str): The text to embed.
        embedding_model_str (str): The embedding model to use in format "provider:model_name".

    Returns:
        list[float]: The computed embedding as a list of floats.
    """
    embedding_model = create_embeddings_model(embedding_model_str)
    embeddings = embedding_model.get_embedding(text)
    return embeddings


def process_file_and_store(
    file_path: str,
    agent_id: str,
    embedding_model: str,
    document_id: str | None = None,
    file_size_bytes: int | None = None,
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

    Returns:
        tuple[bool, str]: (Success status, Document ID)
    """
    from src.models.rag import Document

    logger.info(f"Starting file processing for: {file_path} with agent_id: {agent_id}")

    # Verify file exists
    if not os.path.exists(file_path):
        logger.error(f"File '{file_path}' does not exist.")
        return False, ""

    # Check if file type is supported by scraper
    if not scraper_service.is_supported_file(file_path):
        logger.error(f"Unsupported file type: {os.path.splitext(file_path)[1]}")
        return False, ""

    # Use the file's basename as the document name
    document_name = os.path.basename(file_path)

    # Compute file size if not provided
    if file_size_bytes is None:
        file_size_bytes = os.path.getsize(file_path)

    # Get database instances
    document_dao = get_document_dao()
    context_dao = get_context_dao()

    try:
        # Use scraper service to extract and chunk the document
        logger.info(f"Scraping file with ScraperService: {file_path}")
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

        # Process and store each chunk
        total_chunks = len(scraped_documents)
        for chunk_idx, scraped_doc in enumerate(scraped_documents):
            try:
                # Compute embedding for this chunk using the agent's embedding model
                try:
                    embedding = compute_embedding(scraped_doc.content, embedding_model)
                except (EmbeddingAPIError, EmbeddingError):
                    # Re-raise embedding-specific errors so they can be handled properly
                    raise
                except ValueError as e:
                    # Invalid embedding model format
                    logger.error(f"Invalid embedding model format for chunk {chunk_idx}: {e}")
                    raise EmbeddingError(f"Invalid embedding model configuration: {e!s}", e) from e
                
                # Store context for this chunk
                context_dao.insert_context(
                    document_id=document_id,
                    agent_id=agent_id,
                    embedding=embedding,
                    context=Context(
                        text=scraped_doc.content,
                        document_name=document_name,
                        document_id=document_id,
                        chunk_id=scraped_doc.document_id,  # Use the scraped doc's unique ID as chunk_id
                        chunk_index=chunk_idx,
                        total_chunks=total_chunks,
                    ),
                )
                logger.debug(f"Stored chunk {chunk_idx + 1}/{total_chunks} for document '{document_id}'")
                
            except (EmbeddingAPIError, EmbeddingError):
                # Re-raise embedding errors to be handled at a higher level
                raise
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_idx} of '{file_path}': {e}")
                # Continue processing other chunks
                continue

        logger.info(f"Successfully stored '{document_name}' with {total_chunks} chunks into the database.")

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
