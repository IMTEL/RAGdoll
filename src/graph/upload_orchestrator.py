"""
Upload orchestration service for document processing.

This service handles the coordination of file uploads, document processing,
and streaming responses for different processing strategies.
"""

import asyncio
import io
import json
import logging
import time
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import httpx
from fastapi import HTTPException, Request, UploadFile

from src.core.config import settings, Config
from src.db.content.dao import PostgresDatabase
from src.services.chunk_processing import ChunkProcessingService, ChunkProcessingResult
from src.services.embedding.base import EmbeddingModel
from src.services.knowledge_graph.knowledge_graph_service import KnowledgeGraphService
from src.utils.performance import PerformanceTimer, PerformanceTracker, create_performance_summary

logger = logging.getLogger(__name__)


class UploadOrchestrator:
    """
    Orchestrates the upload and processing of documents.
    
    This service handles the complete pipeline from file upload to knowledge graph
    population, supporting different processing strategies (streaming, batch, parallel).
    """
    
    def __init__(
        self,
        dao: PostgresDatabase,
        embedding_model: EmbeddingModel,
        kg_service: KnowledgeGraphService,
        httpx_client: httpx.Client
    ):
        """
        Initialize the upload orchestrator.
        
        Args:
            dao: Database access object
            embedding_model: Embedding model for text vectorization
            kg_service: Knowledge graph service
            httpx_client: HTTP client for scraper communication
        """
        self.dao = dao
        self.embedding_model = embedding_model
        self.kg_service = kg_service
        self.httpx_client = httpx_client
        self.chunk_processor = ChunkProcessingService(dao, embedding_model, kg_service)
        
    async def setup_upload_session(
        self, 
        files: List[UploadFile], 
        graph_id: Optional[UUID] = None
    ) -> Tuple[UUID, List[UUID], List[Tuple[str, bytes, str]]]:
        """
        Set up a new upload session with graph and document metadata.
        
        Args:
            files: List of uploaded files
            graph_id: Optional existing graph ID
            
        Returns:
            Tuple of (graph_id, document_ids, buffered_files)
        """
        setup_timings = {}
        
        # Step 1: Graph metadata setup
        with PerformanceTimer("Graph metadata setup") as timer:
            if graph_id is None:
                name = " & ".join(f.filename for f in files)[:50] + "…"
                meta = self.dao.post_graph_meta({"name": name})
                if not meta:
                    raise HTTPException(500, "Could not create graph metadata")
                graph_id = meta.id
            else:
                if self.dao.get_graph_meta(graph_id) is None:
                    raise HTTPException(404, f"Graph {graph_id} not found")

            self.kg_service.init_with_graph_id(graph_id)
            if self.kg_service.graph_id is None:
                raise HTTPException(500, "Failed to initialize knowledge graph service")
        setup_timings["graph_metadata"] = timer.duration
        
        # Step 2: Document persistence
        with PerformanceTimer("Document persistence") as timer:
            doc_ids = []
            for f in files:
                doc = self.dao.post_document(f.filename)
                if not doc:
                    raise HTTPException(500, f"Failed to persist document {f.filename}")
                doc_ids.append(doc.id)
        setup_timings["document_persistence"] = timer.duration
        
        # Step 3: Buffer files
        with PerformanceTimer("File buffering") as timer:
            buffered_files = []
            for f in files:
                f.file.seek(0)
                data = f.file.read()
                buffered_files.append((f.filename, data, f.content_type))
        setup_timings["file_buffering"] = timer.duration
        
        logger.info(f"Upload session setup completed in {sum(setup_timings.values()):.4f}s")
        
        return graph_id, doc_ids, buffered_files
    
    def _build_scraper_payload(
        self, 
        buffered_files: List[Tuple[str, bytes, str]], 
        doc_ids: List[UUID]
    ) -> List[Tuple[str, Tuple]]:
        """
        Build multipart payload for scraper service.
        
        Args:
            buffered_files: List of (filename, data, content_type) tuples
            doc_ids: List of document IDs
            
        Returns:
            Multipart payload for HTTP request
        """
        multipart_payload = []
        
        for filename, data, content_type in buffered_files:
            multipart_payload.append(
                ("files", (filename, io.BytesIO(data), content_type))
            )
        
        for doc_id in doc_ids:
            multipart_payload.append(("uuids", (None, str(doc_id))))
            
        return multipart_payload
    
    def _get_auth_headers(self, request: Request) -> Dict[str, str]:
        """
        Extract authentication headers from request.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Dictionary of headers including auth if present
        """
        headers = {"Accept": "application/json"}
        if auth := request.headers.get("authorization"):
            headers["Authorization"] = auth
        return headers
    
    async def stream_processing_results(
        self,
        request: Request,
        graph_id: UUID,
        buffered_files: List[Tuple[str, bytes, str]],
        doc_ids: List[UUID],
        concurrency_limit: int = None
    ):
        """
        Stream processing results for parallel chunk processing.
        
        Args:
            request: FastAPI request object
            graph_id: Graph ID for knowledge graph
            buffered_files: Buffered file data
            doc_ids: Document IDs
            concurrency_limit: Maximum concurrent processing tasks
            
        Yields:
            JSON-encoded processing results
        """
        if concurrency_limit is None:
            concurrency_limit = Config.PARALLEL_CHUNK_LIMIT
            
        request_start = time.perf_counter()
        performance_tracker = PerformanceTracker()
        
        # Statistics tracking
        total_chunks = 0
        successful_chunks = 0
        failed_chunks = 0
        
        # Build scraper request
        multipart_payload = self._build_scraper_payload(buffered_files, doc_ids)
        headers = self._get_auth_headers(request)
        scraper_url = f"{settings.SCRAPER_SERVICE_URL}/stream/upload/"
        
        # Create result queue for processed chunks
        result_queue = asyncio.Queue()
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrency_limit)
        
        async def process_chunk_with_semaphore(chunk_data: Dict):
            """Process a single chunk with concurrency control."""
            nonlocal successful_chunks, failed_chunks
            
            async with semaphore:
                result = await self.chunk_processor.process_chunk(chunk_data)
                
                # Update statistics
                if result.status == "success":
                    successful_chunks += 1
                elif result.status == "partial_success":
                    successful_chunks += 1
                else:
                    failed_chunks += 1
                
                # Track performance
                performance_tracker.merge_timings(result.performance_timings)
                
                # Put result in queue
                await result_queue.put(result.to_dict())
        
        async def process_scraper_stream():
            """Process the scraper stream and queue chunk processing tasks."""
            nonlocal total_chunks
            
            tasks = []
            scraper_start = time.perf_counter()
            
            with self.httpx_client.stream("POST", scraper_url, files=multipart_payload, headers=headers) as resp:
                if resp.status_code != 200:
                    body = resp.read()
                    detail = body.decode("utf-8", "ignore")
                    raise HTTPException(resp.status_code, f"Scraper error: {detail}")
                
                first_chunk_time = None
                for raw_line in resp.iter_lines():
                    if first_chunk_time is None:
                        first_chunk_time = time.perf_counter()
                        logger.info(f"⏱ Time to first chunk: {first_chunk_time - scraper_start:.4f}s")
                    
                    line = raw_line.strip()
                    if not line or line == "[DONE]":
                        if line == "[DONE]":
                            scraper_end = time.perf_counter()
                            scraper_time = scraper_end - scraper_start
                            logger.info(f"⏱ Scraper streaming time: {scraper_time:.4f}s")
                            break
                        continue
                    
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        logger.info(f"Skipping invalid JSON: {line}")
                        continue
                    
                    if "error" in chunk or "text" not in chunk:
                        logger.info(f"Skipping invalid chunk: {line}")
                        continue
                    
                    total_chunks += 1
                    logger.info(f"Queuing chunk {total_chunks}: {chunk['text'][:50]}...")
                    
                    # Create task for processing this chunk
                    task = asyncio.create_task(process_chunk_with_semaphore(chunk))
                    tasks.append(task)
            
            # Wait for all tasks to complete
            await asyncio.gather(*tasks)
            
            # Signal completion
            await result_queue.put({"status": "completed", "total_chunks": total_chunks})
        
        # Start processing task
        processing_task = asyncio.create_task(process_scraper_stream())
        
        # Yield results as they become available
        while True:
            try:
                result = await asyncio.wait_for(result_queue.get(), timeout=1.0)
                
                if result.get("status") == "completed":
                    # Generate final summary
                    request_end = time.perf_counter()
                    total_time = request_end - request_start
                    
                    summary = create_performance_summary(
                        total_time=total_time,
                        chunk_count=total_chunks,
                        tracker=performance_tracker,
                        additional_metrics={
                            "successful_chunks": successful_chunks,
                            "failed_chunks": failed_chunks,
                            "success_rate": successful_chunks / total_chunks if total_chunks > 0 else 0
                        }
                    )
                    
                    yield json.dumps({
                        "status": "processing_complete",
                        "graph_id": str(graph_id),
                        "message": f"Processing complete: {successful_chunks}/{total_chunks} chunks processed successfully",
                        "statistics": {
                            "total_chunks": total_chunks,
                            "successful_chunks": successful_chunks,
                            "failed_chunks": failed_chunks
                        },
                        "performance_summary": summary
                    }) + "\n"
                    break
                else:
                    yield json.dumps(result) + "\n"
                    
            except asyncio.TimeoutError:
                # Check if processing is still ongoing
                if processing_task.done():
                    break
                continue
            except Exception as e:
                logger.error(f"Error in result stream: {e}")
                yield json.dumps({
                    "status": "error",
                    "message": f"Error in result stream: {e}"
                }) + "\n"
                break
    
    async def batch_processing_results(
        self,
        request: Request,
        graph_id: UUID,
        buffered_files: List[Tuple[str, bytes, str]],
        doc_ids: List[UUID],
        batch_size: int = None
    ):
        """
        Stream processing results for batch chunk processing.
        
        Args:
            request: FastAPI request object
            graph_id: Graph ID for knowledge graph
            buffered_files: Buffered file data
            doc_ids: Document IDs
            batch_size: Size of processing batches
            
        Yields:
            JSON-encoded processing results
        """
        if batch_size is None:
            batch_size = Config.BATCH_SIZE
            
        start_time = time.perf_counter()
        performance_tracker = PerformanceTracker()
        
        # Statistics tracking
        total_chunks = 0
        successful_chunks = 0
        failed_chunks = 0
        
        # Build scraper request
        multipart_payload = self._build_scraper_payload(buffered_files, doc_ids)
        headers = self._get_auth_headers(request)
        scraper_url = f"{settings.SCRAPER_SERVICE_URL}/stream/upload/"
        
        async def process_chunk_batch(chunks: List[Dict]):
            """Process a batch of chunks."""
            nonlocal successful_chunks, failed_chunks
            
            if not chunks:
                return
            
            batch_start = time.perf_counter()
            
            try:
                # Use batch processing for better performance
                results = await self.chunk_processor.process_chunks_batch(chunks, batch_size)
                
                # Update statistics and yield results
                for result in results:
                    if result.status == "success":
                        successful_chunks += 1
                    elif result.status == "partial_success":
                        successful_chunks += 1
                    else:
                        failed_chunks += 1
                    
                    # Track performance
                    performance_tracker.merge_timings(result.performance_timings)
                    
                    # Yield result
                    yield json.dumps(result.to_dict()) + "\n"
                
                batch_end = time.perf_counter()
                batch_time = batch_end - batch_start
                performance_tracker.add_timing("batch_processing", batch_time)
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                # Fall back to individual processing
                for chunk in chunks:
                    result = await self.chunk_processor.process_chunk(chunk)
                    
                    if result.status in ["success", "partial_success"]:
                        successful_chunks += 1
                    else:
                        failed_chunks += 1
                    
                    performance_tracker.merge_timings(result.performance_timings)
                    yield json.dumps(result.to_dict()) + "\n"
        
        # Process scraper stream
        chunk_buffer = []
        scraper_start = time.perf_counter()
        
        with self.httpx_client.stream("POST", scraper_url, files=multipart_payload, headers=headers) as resp:
            if resp.status_code != 200:
                body = resp.read()
                detail = body.decode("utf-8", "ignore")
                raise HTTPException(resp.status_code, f"Scraper error: {detail}")
            
            first_chunk_time = None
            for raw_line in resp.iter_lines():
                line = raw_line.strip()
                if not line:
                    continue
                    
                if line == "[DONE]":
                    # Process remaining chunks
                    if chunk_buffer:
                        async for result in process_chunk_batch(chunk_buffer):
                            yield result
                    
                    scraper_end = time.perf_counter()
                    scraper_time = scraper_end - scraper_start
                    logger.info(f"⏱️ Scraper streaming time (batch): {scraper_time:.4f}s")
                    break
                
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter()
                    logger.info(f"⏱️ Time to first chunk (batch): {first_chunk_time - scraper_start:.4f}s")
                
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                if "error" in chunk or "text" not in chunk:
                    continue
                
                total_chunks += 1
                chunk_buffer.append(chunk)
                
                # Process batch when buffer is full
                if len(chunk_buffer) >= batch_size:
                    async for result in process_chunk_batch(chunk_buffer):
                        yield result
                    chunk_buffer = []
        
        # Generate final summary
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        summary = create_performance_summary(
            total_time=total_time,
            chunk_count=total_chunks,
            tracker=performance_tracker,
            additional_metrics={
                "successful_chunks": successful_chunks,
                "failed_chunks": failed_chunks,
                "success_rate": successful_chunks / total_chunks if total_chunks > 0 else 0
            }
        )
        
        yield json.dumps({
            "status": "processing_complete",
            "graph_id": str(graph_id),
            "message": f"Batch processing complete: {successful_chunks}/{total_chunks} chunks processed successfully",
            "statistics": {
                "total_chunks": total_chunks,
                "successful_chunks": successful_chunks,
                "failed_chunks": failed_chunks,
            },
            "performance_summary": summary
        }) + "\n"
