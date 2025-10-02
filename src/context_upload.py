import logging
import os
import uuid

from src.config import Config
from src.rag_service.dao import get_database
from src.rag_service.embeddings import create_embeddings_model


# Load configuration and initialize the SentenceTransformer model.
config = Config()
embedding_model = create_embeddings_model()


def compute_embedding(text: str) -> list[float]:
    """Computes an embedding for the given text using a SentenceTransformer model.

    Args:
        text (str): The text to embed.

    Returns:
        list[float]: The computed embedding as a list of floats.
    """
    embeddings = embedding_model.get_embedding(text)
    return embeddings


def process_file_and_store(file_path: str, category: str) -> bool:
    """Processes a .txt or .md file, extracts its text, computes its embedding, and stores the data in the database.

    Args:
        file_path (str): Path to the text or markdown file.
        category (str): Document category to associate with this context.

    Returns:
        bool: True if storing was successful; False otherwise.
    """
    logging.info(f"Starting file processing for: {file_path} with category: {category}")

    # Verify file exists
    if not os.path.exists(file_path):
        logging.error(f"File '{file_path}' does not exist.")
        return False

    # Verify file extension is supported.
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in [".txt", ".md"]:
        logging.error("Unsupported file type. Only .txt and .md files are supported.")
        return False

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
            logging.error(f"Error reading file '{file_path}': {e}")
            return False
    except Exception as e:
        logging.error(f"Error reading file '{file_path}': {e}")
        return False

    # Compute the embedding using the actual model.
    try:
        embedding = compute_embedding(text)
    except Exception as e:
        logging.error(f"Error computing embedding for file '{file_path}': {e}")
        return False

    # Use the file's basename as the document name.
    document_name = os.path.basename(file_path)

    # Generate a unique document ID.
    # TODO: Is this really a unique enough ID?
    document_id = str(uuid.uuid4())

    # Get the configured database instance.
    db = get_database()

    try:
        # Changed from NPC parameter to category parameter
        success = db.post_context(
            text=text,
            category=category,  # Using category instead of NPC
            embedding=embedding,
            document_id=document_id,
            document_name=document_name,
        )
    except Exception as e:
        logging.error(f"Error posting context to the database: {e}")
        return False

    if success:
        logging.info(
            f"Successfully stored '{document_name}' into the database with category '{category}'."
        )
    else:
        logging.error(f"Failed to store '{document_name}' into the database.")

    return success


if __name__ == "__main__":
    # Example usage of the process_file_and_store function.
    file_path = "src.context_files.salmon.txt"
    category = "General Information"  # Changed from NPC to category
    process_file_and_store(file_path, category)
