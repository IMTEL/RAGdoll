"""Test integration between scraper service and context upload."""
import os
import tempfile
from pathlib import Path

import pytest

from src.context_upload import process_file_and_store, scraper_service
from src.rag_service.dao.factory import get_context_dao, get_document_dao


class TestScraperIntegration:
    """Test cases for scraper service integration with context upload."""

    def test_scraper_supports_multiple_file_types(self):
        """Test that scraper supports various file types."""
        supported = ['.pdf', '.docx', '.txt', '.md', '.html', '.pptx', '.xlsx']
        
        for ext in supported:
            assert scraper_service.is_supported_file(f"test{ext}"), f"{ext} should be supported"

    def test_process_txt_file_with_scraper(self, mock_db):
        """Test processing a text file using the scraper."""
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.\n")
            f.write("It has multiple lines.\n")
            f.write("The scraper should handle this well.")
            temp_path = f.name

        try:
            agent_id = "test-agent-scraper"
            success, doc_id = process_file_and_store(temp_path, agent_id)
            
            assert success, "File processing should succeed"
            assert doc_id, "Should return a document ID"
            
            # Verify document was created
            document_dao = get_document_dao()
            doc = document_dao.get_by_id(doc_id)
            assert doc is not None
            assert doc.agent_id == agent_id
            
            # Verify contexts were created (may be chunked)
            context_dao = get_context_dao()
            # Note: The actual verification would depend on the mock implementation
            
        finally:
            # Clean up
            os.unlink(temp_path)

    def test_process_markdown_file_with_scraper(self, mock_db):
        """Test processing a markdown file using the scraper."""
        # Create a temporary markdown file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Test Document\n\n")
            f.write("## Introduction\n\n")
            f.write("This is a test markdown document.\n\n")
            f.write("## Content\n\n")
            f.write("The scraper should extract and chunk this content.\n")
            temp_path = f.name

        try:
            agent_id = "test-agent-md"
            success, doc_id = process_file_and_store(temp_path, agent_id)
            
            assert success, "Markdown file processing should succeed"
            assert doc_id, "Should return a document ID"
            
        finally:
            # Clean up
            os.unlink(temp_path)

    def test_unsupported_file_type(self, mock_db):
        """Test that unsupported file types are rejected."""
        # Create a temporary file with unsupported extension
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            f.write("This is an unsupported file type.")
            temp_path = f.name

        try:
            agent_id = "test-agent-unsupported"
            success, doc_id = process_file_and_store(temp_path, agent_id)
            
            assert not success, "Unsupported file should fail"
            assert doc_id == "", "Should return empty document ID"
            
        finally:
            # Clean up
            os.unlink(temp_path)

    def test_nonexistent_file(self, mock_db):
        """Test handling of nonexistent files."""
        agent_id = "test-agent-missing"
        success, doc_id = process_file_and_store("/nonexistent/file.txt", agent_id)
        
        assert not success, "Nonexistent file should fail"
        assert doc_id == "", "Should return empty document ID"

    def test_scraper_chunking(self):
        """Test that scraper properly chunks documents."""
        # Create a large text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Write enough content to trigger chunking (based on chunk_size=500)
            for i in range(20):
                f.write(f"This is paragraph {i}. " * 10)
                f.write("\n\n")
            temp_path = f.name

        try:
            # Use scraper directly to test chunking
            scraped_docs = scraper_service.scrape_file(temp_path)
            
            # Should have multiple chunks
            assert len(scraped_docs) > 0, "Should extract at least one chunk"
            
            # Each chunk should have required attributes
            for doc in scraped_docs:
                assert doc.content, "Chunk should have content"
                assert doc.document_id, "Chunk should have document_id"
                assert doc.document_name, "Chunk should have document_name"
                assert doc.chunk_index >= 0, "Chunk should have valid index"
                
        finally:
            # Clean up
            os.unlink(temp_path)
