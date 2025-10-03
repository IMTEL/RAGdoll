import argparse
import logging
import os
import sys

from src.context_upload import process_file_and_store
from src.rag_service.embeddings import create_embeddings_model
from src.rag_service.repositories import get_context_repository


# Add the project root to Python's path
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.append(PROJECT_ROOT)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize embedding model
embedding_model = create_embeddings_model()


def upload_document(file_path, category="General Information"):
    """Upload a document to the mock database."""
    success = process_file_and_store(file_path, category)
    return success


def list_documents():
    """List all documents in the mock database."""
    db = get_context_repository()
    if not hasattr(db, "data"):
        logger.error("Database doesn't have data attribute. Is it a MockDatabase?")
        return []

    documents = []
    for doc in db.data:
        documents.append(
            {
                "document_id": doc.get("document_id", "N/A"),
                "document_name": doc.get("document_name", "N/A"),
                "category": doc.get(
                    "category", doc.get("npc", "N/A")
                ),  # Handle both new and old formats
                "text_preview": doc.get("text", "")[:50] + "..."
                if len(doc.get("text", "")) > 50
                else doc.get("text", ""),
            }
        )
    return documents


def get_document_by_id(document_id):
    """Get a document by its ID."""
    db = get_context_repository()
    if not hasattr(db, "data"):
        logger.error("Database doesn't have data attribute. Is it a MockDatabase?")
        return None

    for doc in db.data:
        if doc.get("document_id") == document_id:
            return doc
    return None


def get_document_by_name(document_name):
    """Get documents by name."""
    db = get_context_repository()
    if not hasattr(db, "data"):
        logger.error("Database doesn't have data attribute. Is it a MockDatabase?")
        return []

    results = []
    for doc in db.data:
        if doc.get("document_name") == document_name:
            results.append(doc)
    return results


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Mock Database Client for Chat Service"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload a document")
    upload_parser.add_argument("file", help="Path to the file to upload")
    upload_parser.add_argument(
        "--category",
        default="General Information",
        help="Document category (default: General Information)",
    )

    # List command
    subparsers.add_parser("list", help="List all documents")

    # Get command
    get_parser = subparsers.add_parser("get", help="Get document details")
    get_group = get_parser.add_mutually_exclusive_group(required=True)
    get_group.add_argument("--id", help="Document ID")
    get_group.add_argument("--name", help="Document name")

    # Parse arguments
    args = parser.parse_args()

    if args.command == "upload":
        # Upload a document
        if not os.path.exists(args.file):
            logger.error(f"File not found: {args.file}")
            return 1

        success = upload_document(args.file, args.category)
        if success:
            logger.info(f"Successfully uploaded {args.file}")
        else:
            logger.error(f"Failed to upload {args.file}")
            return 1

    elif args.command == "list":
        # List all documents
        documents = list_documents()
        if not documents:
            logger.info("No documents found in the database")
        else:
            logger.info(f"Found {len(documents)} documents:")
            for doc in documents:
                logger.info(
                    f"ID: {doc['document_id']}, Name: {doc['document_name']}, Category: {doc['category']}"
                )
                logger.info(f"Preview: {doc['text_preview']}")
                logger.info("-" * 40)

    elif args.command == "get":
        # Get document details
        if args.id:
            doc = get_document_by_id(args.id)
            if doc:
                logger.info(f"Document ID: {doc.get('document_id')}")
                logger.info(f"Document Name: {doc.get('document_name')}")
                logger.info(f"Category: {doc.get('category', doc.get('npc', 'N/A'))}")
                logger.info(f"Text: {doc.get('text')[:500]}...")
            else:
                logger.error(f"Document with ID {args.id} not found")
                return 1

        elif args.name:
            docs = get_document_by_name(args.name)
            if docs:
                logger.info(f"Found {len(docs)} documents with name {args.name}:")
                for doc in docs:
                    logger.info(f"Document ID: {doc.get('document_id')}")
                    logger.info(
                        f"Category: {doc.get('category', doc.get('npc', 'N/A'))}"
                    )
                    logger.info(f"Text: {doc.get('text')[:500]}...")
                    logger.info("-" * 40)
            else:
                logger.error(f"No documents found with name {args.name}")
                return 1

    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())
