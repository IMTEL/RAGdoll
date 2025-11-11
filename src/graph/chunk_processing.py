"""
Chunk processing service for document ingestion.

This service handles the processing of individual text chunks including:
- Text sanitization
- Embedding generation
- Database persistence
- Knowledge graph population
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Union
from uuid import UUID

from src.db.content.dao import PostgresDatabase
from src.services.embedding.base import EmbeddingModel
from src.services.knowledge_graph.knowledge_graph_service import KnowledgeGraphService
from src.utils.performance import PerformanceTimer, PerformanceTracker
from src.utils.text_sanitizer import sanitize_text

logger = logging.getLogger(__name__)


class ChunkProcessingResult:
    """
    Result of processing a single chunk.
    
    Attributes:
        status: Processing status ('success', 'partial_success', 'failed')
        chunk_id: ID of the saved chunk (if successful)
        document_id: ID of the source document
        page_num: Page number in the document
        chunk_index: Index of the chunk on the page
        text: Processed text content (truncated for logging)
        error: Error message (if failed)
        performance_timings: Timing data for each operation
    """
    
    def __init__(
        self,
        status: str,
        document_id: str,
        page_num: int,
        chunk_index: int,
        text: str,
        chunk_id: Optional[str] = None,
        error: Optional[str] = None,
        performance_timings: Optional[Dict[str, float]] = None
    ):
        self.status = status
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.page_num = page_num
        self.chunk_index = chunk_index
        self.text = text[:100] + "..." if len(text) > 100 else text
        self.error = error
        self.performance_timings = performance_timings or {}
        
    def to_dict(self) -> Dict:
        """Convert result to dictionary for JSON serialization."""
        result = {
            "status": self.status,
            "document_id": self.document_id,
            "page_num": self.page_num,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "processed_at": str(asyncio.get_event_loop().time()),
            "performance_timings": self.performance_timings
        }
        
        if self.chunk_id:
            result["chunk_id"] = self.chunk_id
            
        if self.error:
            result["error"] = self.error
            
        return result


class ChunkProcessingService:
    """
    Service for processing document chunks asynchronously.
    
    This service handles the complete pipeline for processing a text chunk:
    1. Text sanitization
    2. Embedding generation
    3. Database persistence
    4. Knowledge graph population
    """
    
    def __init__(
        self,
        dao: PostgresDatabase,
        embedding_model: EmbeddingModel,
        kg_service: KnowledgeGraphService,
        max_retries: int = 3
    ):
        """
        Initialize the chunk processing service.
        
        Args:
            dao: Database access object
            embedding_model: Embedding model for text vectorization
            kg_service: Knowledge graph service
            max_retries: Maximum number of retry attempts for failed operations
        """
        self.dao = dao
        self.embedding_model = embedding_model
        self.kg_service = kg_service
        self.max_retries = max_retries
        
    async def process_chunk(self, chunk_data: Dict) -> ChunkProcessingResult:
        """
        Process a single chunk asynchronously.
        
        Args:
            chunk_data: Dictionary containing chunk information
            
        Returns:
            ChunkProcessingResult with processing status and metrics
        """
        total_start = time.perf_counter()
        
        # Extract chunk information
        text = chunk_data["text"]
        document_id = str(chunk_data["document_id"])
        page_num = chunk_data["page_num"]
        chunk_index = chunk_data["chunk_index"]
        
        timings = {
            "text_sanitization": 0,
            "embedding_generation": 0,
            "database_save": 0,
            "kg_population": 0,
            "total_processing": 0
        }
        
        try:
            # Step 1: Sanitize text
            with PerformanceTimer("Text sanitization") as timer:
                sanitized_text = sanitize_text(text)
            timings["text_sanitization"] = timer.duration
            
            # Step 2: Generate embedding with retry logic
            saved_chunk = await self._save_chunk_with_retry(
                sanitized_text, document_id, page_num, chunk_index, timings
            )
            
            if not saved_chunk:
                total_end = time.perf_counter()
                timings["total_processing"] = total_end - total_start
                
                return ChunkProcessingResult(
                    status="failed",
                    document_id=document_id,
                    page_num=page_num,
                    chunk_index=chunk_index,
                    text=sanitized_text,
                    error="Failed to save chunk to database after retries",
                    performance_timings=timings
                )
            
            # Step 3: Knowledge graph population
            try:
                with PerformanceTimer("Knowledge graph population") as timer:
                    self.kg_service.populate_graph_from_text(
                        sanitized_text,
                        str(saved_chunk.id),
                        str(saved_chunk.document_id)
                    )
                timings["kg_population"] = timer.duration
                
                # Success
                total_end = time.perf_counter()
                timings["total_processing"] = total_end - total_start
                
                logger.info(f"Successfully processed chunk: {sanitized_text[:50]}...")
                
                return ChunkProcessingResult(
                    status="success",
                    chunk_id=str(saved_chunk.id),
                    document_id=document_id,
                    page_num=page_num,
                    chunk_index=chunk_index,
                    text=sanitized_text,
                    performance_timings=timings
                )
                
            except Exception as e:
                # Partial success - chunk saved but KG failed
                total_end = time.perf_counter()
                timings["total_processing"] = total_end - total_start
                
                logger.error(f"KG ingestion failed for chunk {saved_chunk.id}: {e}")
                
                return ChunkProcessingResult(
                    status="partial_success",
                    chunk_id=str(saved_chunk.id),
                    document_id=document_id,
                    page_num=page_num,
                    chunk_index=chunk_index,
                    text=sanitized_text,
                    error=str(e),
                    performance_timings=timings
                )
                
        except Exception as e:
            # Complete failure
            total_end = time.perf_counter()
            timings["total_processing"] = total_end - total_start
            
            logger.error(f"Unexpected error processing chunk: {e}")
            
            return ChunkProcessingResult(
                status="failed",
                document_id=document_id,
                page_num=page_num,
                chunk_index=chunk_index,
                text=sanitized_text if 'sanitized_text' in locals() else text,
                error=str(e),
                performance_timings=timings
            )
    
    async def _save_chunk_with_retry(
        self, 
        text: str, 
        document_id: str, 
        page_num: int, 
        chunk_index: int,
        timings: Dict[str, float]
    ):
        """
        Save chunk to database with retry logic.
        
        Args:
            text: Sanitized text content
            document_id: ID of the source document
            page_num: Page number
            chunk_index: Chunk index
            timings: Dictionary to store timing information
            
        Returns:
            Saved chunk object or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                # Generate embedding
                with PerformanceTimer(f"Embedding generation (attempt {attempt + 1})") as timer:
                    embedding = await self.embedding_model.get_embedding(text)
                timings["embedding_generation"] = timer.duration
                
                # Save to database
                with PerformanceTimer(f"Database save (attempt {attempt + 1})") as timer:
                    saved = self.dao.post_chunk({
                        "text": text,
                        "embedding": embedding,
                        "document_id": document_id,
                        "page_num": page_num,
                        "chunk_index": chunk_index,
                    })
                timings["database_save"] = timer.duration
                
                if saved:
                    return saved
                    
                logger.warning(f"Failed to persist chunk (attempt {attempt + 1}/{self.max_retries})")
                
            except Exception as e:
                logger.warning(f"Error persisting chunk (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to persist chunk after {self.max_retries} attempts")
                    
        return None
    
    async def process_chunks_parallel(
        self, 
        chunks: List[Dict], 
        concurrency_limit: int = 10
    ) -> List[ChunkProcessingResult]:
        """
        Process multiple chunks in parallel with concurrency control.
        
        Args:
            chunks: List of chunk data dictionaries
            concurrency_limit: Maximum number of concurrent processing tasks
            
        Returns:
            List of ChunkProcessingResult objects
        """
        semaphore = asyncio.Semaphore(concurrency_limit)
        
        async def process_with_semaphore(chunk_data: Dict) -> ChunkProcessingResult:
            async with semaphore:
                return await self.process_chunk(chunk_data)
        
        tasks = [process_with_semaphore(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def process_chunks_batch(
        self, 
        chunks: List[Dict], 
        batch_size: int = 5
    ) -> List[ChunkProcessingResult]:
        """
        Process chunks in batches with optimized embedding generation.
        
        Args:
            chunks: List of chunk data dictionaries
            batch_size: Number of chunks to process in each batch
            
        Returns:
            List of ChunkProcessingResult objects
        """
        results = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            try:
                # Batch embedding generation
                texts = [sanitize_text(chunk["text"]) for chunk in batch]
                embeddings = await self.embedding_model.get_embeddings_batch(texts)
                
                # Process each chunk in the batch
                batch_results = []
                for chunk_data, embedding in zip(batch, embeddings):
                    result = await self._process_chunk_with_embedding(chunk_data, embedding)
                    batch_results.append(result)
                
                results.extend(batch_results)
                
            except Exception as e:
                # Fall back to individual processing
                logger.warning(f"Batch processing failed: {e}. Falling back to individual processing.")
                for chunk_data in batch:
                    result = await self.process_chunk(chunk_data)
                    results.append(result)
        
        return results
    
    async def _process_chunk_with_embedding(
        self, 
        chunk_data: Dict, 
        embedding: List[float]
    ) -> ChunkProcessingResult:
        """
        Process a chunk with a pre-generated embedding.
        
        Args:
            chunk_data: Dictionary containing chunk information
            embedding: Pre-generated embedding vector
            
        Returns:
            ChunkProcessingResult with processing status and metrics
        """
        total_start = time.perf_counter()
        
        # Extract chunk information
        text = chunk_data["text"]
        document_id = str(chunk_data["document_id"])
        page_num = chunk_data["page_num"]
        chunk_index = chunk_data["chunk_index"]
        
        timings = {
            "text_sanitization": 0,
            "embedding_generation": 0,  # Already done in batch
            "database_save": 0,
            "kg_population": 0,
            "total_processing": 0
        }
        
        try:
            # Sanitize text
            with PerformanceTimer("Text sanitization (batch)") as timer:
                sanitized_text = sanitize_text(text)
            timings["text_sanitization"] = timer.duration
            
            # Save to database
            with PerformanceTimer("Database save (batch)") as timer:
                saved = self.dao.post_chunk({
                    "text": sanitized_text,
                    "embedding": embedding,
                    "document_id": document_id,
                    "page_num": page_num,
                    "chunk_index": chunk_index,
                })
            timings["database_save"] = timer.duration
            
            if not saved:
                total_end = time.perf_counter()
                timings["total_processing"] = total_end - total_start
                
                return ChunkProcessingResult(
                    status="failed",
                    document_id=document_id,
                    page_num=page_num,
                    chunk_index=chunk_index,
                    text=sanitized_text,
                    error="Failed to save chunk to database",
                    performance_timings=timings
                )
            
            # Knowledge graph population
            try:
                with PerformanceTimer("Knowledge graph population (batch)") as timer:
                    self.kg_service.populate_graph_from_text(
                        sanitized_text,
                        str(saved.id),
                        str(saved.document_id)
                    )
                timings["kg_population"] = timer.duration
                
                total_end = time.perf_counter()
                timings["total_processing"] = total_end - total_start
                
                return ChunkProcessingResult(
                    status="success",
                    chunk_id=str(saved.id),
                    document_id=document_id,
                    page_num=page_num,
                    chunk_index=chunk_index,
                    text=sanitized_text,
                    performance_timings=timings
                )
                
            except Exception as e:
                total_end = time.perf_counter()
                timings["total_processing"] = total_end - total_start
                
                return ChunkProcessingResult(
                    status="partial_success",
                    chunk_id=str(saved.id),
                    document_id=document_id,
                    page_num=page_num,
                    chunk_index=chunk_index,
                    text=sanitized_text,
                    error=str(e),
                    performance_timings=timings
                )
                
        except Exception as e:
            total_end = time.perf_counter()
            timings["total_processing"] = total_end - total_start
            
            return ChunkProcessingResult(
                status="failed",
                document_id=document_id,
                page_num=page_num,
                chunk_index=chunk_index,
                text=sanitized_text if 'sanitized_text' in locals() else text,
                error=str(e),
                performance_timings=timings
            )
