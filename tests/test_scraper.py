from pathlib import Path

import nltk
import pytest

from src.scraper_service.scraper import ScrapedDocument, ScraperService


class TestScraperService:
    """Test cases for the ScraperService class."""

    nltk.download("punkt")
    nltk.download("punkt_tab")

    @pytest.fixture
    def scraper_service(self):
        """Create a ScraperService instance for testing."""
        return ScraperService(chunk_size=800, overlap=100)

    @pytest.fixture
    def test_pdf_path(self):
        """Path to the test PDF file."""
        test_dir = Path(__file__).parent / "test_sets"
        return test_dir / "test_pdf.pdf"

    @pytest.fixture
    def test_docx_path(self):
        """Path to the test DOCX file."""
        test_dir = Path(__file__).parent / "test_sets"
        return test_dir / "test_docs.docx"

    @pytest.mark.unit
    def test_scraper_service_initialization(self, scraper_service):
        """Test that ScraperService initializes correctly."""
        assert scraper_service.chunk_size == 800
        assert scraper_service.overlap == 100
        assert hasattr(scraper_service, "SUPPORTED_EXTENSIONS")

    @pytest.mark.unit
    def test_is_supported_file(self, scraper_service):
        """Test file type support detection."""
        assert scraper_service.is_supported_file("test.pdf")
        assert scraper_service.is_supported_file("test.docx")
        assert scraper_service.is_supported_file("test.txt")
        assert not scraper_service.is_supported_file("test.xyz")

    @pytest.mark.unit
    def test_generate_document_id(self, scraper_service):
        """Test document ID generation."""
        doc_id_1 = scraper_service.generate_document_id("test.pdf", 0)
        doc_id_2 = scraper_service.generate_document_id("test.pdf", 1)
        doc_id_3 = scraper_service.generate_document_id("test.pdf", 0)

        assert (
            doc_id_1 != doc_id_2
        )  # Different chunk indices should generate different IDs
        assert doc_id_1 == doc_id_3  # Same file and chunk should generate same ID
        assert len(doc_id_1) == 32  # MD5 hash length

    @pytest.mark.integration
    def test_get_file_info_pdf(self, scraper_service, test_pdf_path):
        """Test getting file info for PDF file."""
        if not test_pdf_path.exists():
            pytest.skip(f"Test file not found: {test_pdf_path}")

        file_info = scraper_service.get_file_info(str(test_pdf_path))

        assert file_info["file_name"] == "test_pdf.pdf"
        assert file_info["file_type"] == ".pdf"
        assert file_info["is_supported"] is True
        assert file_info["file_size"] > 0
        assert Path(file_info["absolute_path"]).exists()

    @pytest.mark.integration
    def test_scrape_pdf_file(self, scraper_service, test_pdf_path):
        """Test scraping a PDF file."""
        if not test_pdf_path.exists():
            pytest.skip(f"Test file not found: {test_pdf_path}")

        documents = scraper_service.scrape_file(str(test_pdf_path))

        assert isinstance(documents, list)
        assert len(documents) > 0

        # Check first document structure
        first_doc = documents[0]
        assert isinstance(first_doc, ScrapedDocument)
        assert first_doc.content is not None
        assert len(first_doc.content.strip()) > 0
        assert first_doc.document_name == "test_pdf.pdf"
        assert first_doc.file_type == ".pdf"
        assert first_doc.chunk_index == 0
        assert first_doc.source_file == str(test_pdf_path)
        assert isinstance(first_doc.metadata, dict)

    @pytest.mark.unit
    def test_scrape_nonexistent_file(self, scraper_service):
        """Test scraping a file that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            scraper_service.scrape_file("nonexistent_file.pdf")

    @pytest.mark.unit
    def test_scrape_unsupported_file_type(self, scraper_service):
        """Test scraping an unsupported file type."""
        # Create a temporary file with unsupported extension
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as tmp_file:
            tmp_file.write(b"test content")
            tmp_file_path = tmp_file.name

        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                scraper_service.scrape_file(tmp_file_path)
        finally:
            os.unlink(tmp_file_path)

    @pytest.mark.unit
    def test_get_file_info_nonexistent_file(self, scraper_service):
        """Test getting file info for a file that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            scraper_service.get_file_info("nonexistent_file.pdf")
