import os
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.xlsx import partition_xlsx
from unstructured.partition.text import partition_text
from unstructured.partition.html import partition_html
from unstructured.partition.md import partition_md
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element

logger = logging.getLogger(__name__)

@dataclass
class ScrapedDocument:
    """Represents a scraped document with metadata"""
    content: str
    document_id: str
    document_name: str
    file_type: str
    chunk_index: int
    metadata: Dict
    source_file: str

class ScraperService:
    """
    A comprehensive scraper service that can process various file types
    using the unstructured library for RAG applications.
    """
    
    SUPPORTED_EXTENSIONS = {
        '.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls',
        '.txt', '.md', '.html', '.htm', '.xml', '.json', '.csv'
    }
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Initialize the scraper service.
        
        Args:
            chunk_size: Maximum number of characters per chunk
            overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def generate_document_id(self, file_path: str, chunk_index: int = 0) -> str:
        """Generate a unique document ID based on file path and chunk index"""
        file_content = f"{file_path}_{chunk_index}"
        return hashlib.md5(file_content.encode()).hexdigest()
    
    def is_supported_file(self, file_path: str) -> bool:
        """Check if the file type is supported"""
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def extract_elements(self, file_path: str) -> List[Element]:
        """
        Extract elements from a file using unstructured.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            List of unstructured elements
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not self.is_supported_file(file_path):
            raise ValueError(f"Unsupported file type: {Path(file_path).suffix}")
        
        try:
            # Use auto partition for most files, but specific partitioners for better control
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.pdf':
                # Try different strategies for PDF processing
                elements = None
                try:
                    elements = partition_pdf(
                        filename=file_path,
                        strategy="auto",
                        infer_table_structure=True,
                        extract_images_in_pdf=False,  # Set to True if you want image extraction
                        include_page_breaks=True,
                    )
                except Exception as e:
                    logger.warning(f"Failed to process using auto: {str(e)}")

                
                # If all strategies fail, try with OCR
                if not elements or len(elements) == 0:
                    try:
                        logger.info("Trying PDF processing with OCR (ocr_only strategy)")
                        elements = partition_pdf(
                            filename=file_path,
                            strategy="ocr_only",
                            infer_table_structure=True,
                            extract_images_in_pdf=False,
                        )
                        if elements:
                            logger.info(f"Successfully extracted {len(elements)} elements using OCR")
                    except Exception as e:
                        logger.warning(f"OCR strategy also failed: {str(e)}")
                        elements = []
            elif file_extension in ['.docx', '.doc']:
                elements = partition_docx(filename=file_path)
            elif file_extension in ['.pptx', '.ppt']:
                elements = partition_pptx(filename=file_path)
            elif file_extension in ['.xlsx', '.xls']:
                elements = partition_xlsx(filename=file_path)
            elif file_extension == '.html':
                elements = partition_html(filename=file_path)
            elif file_extension == '.md':
                elements = partition_md(filename=file_path)
            elif file_extension == '.txt':
                elements = partition_text(filename=file_path)
            else:
                # Fallback to auto partition
                elements = partition(filename=file_path)
            
            logger.info(f"Extracted {len(elements)} elements from {file_path}")
            
            # Debug: Print information about extracted elements
            if elements:
                logger.info(f"Element types found: {[type(elem).__name__ for elem in elements[:5]]}")
                logger.info(f"Sample element content: {[str(elem)[:100] + '...' if len(str(elem)) > 100 else str(elem) for elem in elements[:2]]}")
            else:
                logger.warning("No elements extracted from the file!")
            
            return elements
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            raise
    
    def chunk_elements(self, elements: List[Element]) -> List[Element]:
        """
        Chunk elements using unstructured's chunking capabilities.
        
        Args:
            elements: List of unstructured elements
            
        Returns:
            List of chunked elements
        """
        try:
            # Use title-based chunking for better semantic coherence
            chunked_elements = chunk_by_title(
                elements,
                max_characters=self.chunk_size,
                overlap=self.overlap,
                combine_text_under_n_chars=50,  # Combine small text elements
            )
            
            logger.info(f"Created {len(chunked_elements)} chunks from {len(elements)} elements")
            
            # Debug: Show content of chunked elements
            if chunked_elements:
                logger.info(f"Sample chunked content: {[str(elem)[:100] + '...' if len(str(elem)) > 100 else str(elem) for elem in chunked_elements[:2]]}")
            
            return chunked_elements
            
        except Exception as e:
            logger.error(f"Error chunking elements: {str(e)}")
            # Fallback to original elements if chunking fails
            return elements
    
    def elements_to_scraped_documents(
        self, 
        elements: List[Element], 
        file_path: str
    ) -> List[ScrapedDocument]:
        """
        Convert unstructured elements to ScrapedDocument objects.
        
        Args:
            elements: List of unstructured elements
            file_path: Original file path
            
        Returns:
            List of ScrapedDocument objects
        """
        documents = []
        file_name = Path(file_path).name
        file_type = Path(file_path).suffix.lower()
        
        for i, element in enumerate(elements):
            # Extract text content
            content = str(element).strip()
            
            # Skip empty content
            if not content:
                continue
            
            # Generate document ID
            doc_id = self.generate_document_id(file_path, i)
            
            # Extract metadata from element
            metadata = {
                'element_type': element.category if hasattr(element, 'category') else 'unknown',
                'page_number': getattr(element.metadata, 'page_number', None) if hasattr(element, 'metadata') else None,
                'coordinates': getattr(element.metadata, 'coordinates', None) if hasattr(element, 'metadata') else None,
                'file_size': os.path.getsize(file_path),
                'processing_timestamp': str(hash(content + file_path)),  # Simple timestamp alternative
            }
            
            # Create ScrapedDocument
            scraped_doc = ScrapedDocument(
                content=content,
                document_id=doc_id,
                document_name=file_name,
                file_type=file_type,
                chunk_index=i,
                metadata=metadata,
                source_file=file_path
            )
            
            documents.append(scraped_doc)
        
        return documents
    
    def scrape_file(self, file_path: str) -> List[ScrapedDocument]:
        """
        Main method to scrape a file and return processed documents.
        
        Args:
            file_path: Path to the file to scrape
            
        Returns:
            List of ScrapedDocument objects ready for RAG processing
        """
        logger.info(f"Starting to scrape file: {file_path}")
        
        try:
            # Extract elements from file
            elements = self.extract_elements(file_path)
            
            # Chunk elements for better RAG performance
            chunked_elements = self.chunk_elements(elements)
            
            # Convert to ScrapedDocument objects
            documents = self.elements_to_scraped_documents(chunked_elements, file_path)
            
            logger.info(f"Successfully scraped {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to scrape file {file_path}: {str(e)}")
            raise
    
    def get_file_info(self, file_path: str) -> Dict:
        """
        Get basic information about a file without processing it.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_stats = os.stat(file_path)
        return {
            'file_name': Path(file_path).name,
            'file_size': file_stats.st_size,
            'file_type': Path(file_path).suffix.lower(),
            'is_supported': self.is_supported_file(file_path),
            'absolute_path': os.path.abspath(file_path)
        }


def test_scraper_with_cv():
    """
    Test function to process the cv.pdf file in the scraper_service directory.
    """
    # Setup logging - use INFO level to avoid spammy debug logs
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Reduce verbosity of third-party libraries
    logging.getLogger('unstructured').setLevel(logging.WARNING)
    logging.getLogger('pdfminer').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # Get the directory of this script
    current_dir = Path(__file__).parent
    cv_file_path = current_dir / "Dsa.pdf"
    
    print(f"Testing scraper with CV file: {cv_file_path}")
    
    # Initialize scraper
    scraper = ScraperService(chunk_size=800, overlap=100)
    
    try:
        # Get file info first
        file_info = scraper.get_file_info(str(cv_file_path))
        print(f"\nFile Info: {file_info}")
        
        # Scrape the CV file
        documents = scraper.scrape_file(str(cv_file_path))
        
        print(f"\nSuccessfully processed CV file!")
        print(f"Number of document chunks created: {len(documents)}")
        
        # Display first few chunks
        for i, doc in enumerate(documents[:3]):  # Show first 3 chunks
            print(f"\n--- Chunk {i+1} ---")
            print(f"Document ID: {doc.document_id}")
            print(f"Content preview: {doc.content[:200]}...")
            print(f"Metadata: {doc.metadata}")
        
        if len(documents) > 3:
            print(f"\n... and {len(documents) - 3} more chunks")
        
        return documents
        
    except Exception as e:
        print(f"Error processing CV file: {str(e)}")
        return None


if __name__ == "__main__":
    test_scraper_with_cv()

