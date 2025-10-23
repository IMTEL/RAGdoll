"""
Scraper client utility for testing and development.

This module provides a utility client for testing the tango-scraper service
integration. It demonstrates how to send files to the scraper service and
stream the extracted chunks back.

Note: This is a development/testing utility and should not be used in production.
"""

import asyncio
import json
import logging
import mimetypes
import os
import pathlib
import uuid
from typing import Iterable

import httpx

logger = logging.getLogger(__name__)

# Configuration
SCRAPER_STREAM_URL = os.getenv(
    "SCRAPER_STREAM_URL",
    "http://scraper-service:8080/stream/upload/",
)
AUTH_HEADER = os.getenv("SCRAPER_BEARER", "")


async def send_files_and_stream_chunks(paths: Iterable[str]) -> None:
    """
    Send files to the scraper service and stream back extracted chunks.
    
    This function demonstrates the integration with the tango-scraper service
    by sending files and processing the streaming response.
    
    Args:
        paths: Iterable of file paths to process
    """
    files = []
    params = []
    handles = []

    # Prepare files for upload
    for p in map(pathlib.Path, paths):
        mime, _ = mimetypes.guess_type(p.name)
        f = p.open("rb")
        handles.append(f)
        files.append(("files", (p.name, f, mime or "application/octet-stream")))
        params.append(("uuids", str(uuid.uuid4())))

    headers = (
        {"Authorization": f"Bearer {AUTH_HEADER}"} if AUTH_HEADER else {}
    )

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                SCRAPER_STREAM_URL,
                params=params,
                files=files,
                headers=headers,
            ) as resp:
                resp.raise_for_status()

                async for line in resp.aiter_lines():
                    if not line:  # Skip keep-alive blank lines
                        continue
                    if line == "[DONE]":
                        logger.info("✓ All chunks received")
                        break

                    try:
                        chunk = json.loads(line)
                        handle_chunk(chunk)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON received: {line}")
                        continue

    finally:
        # Clean up file handles
        for f in handles:
            f.close()


def handle_chunk(chunk: dict) -> None:
    """
    Process a single chunk received from the scraper service.
    
    This is a simple handler that logs chunk information. In a production
    system, this would be replaced with database insertion, message queue
    publishing, or other processing logic.
    
    Args:
        chunk: Dictionary containing chunk data (uuid, page, index, text, etc.)
    """
    logger.info(
        f"Chunk {chunk['uuid']} | page {chunk['page']:>3} | "
        f"idx {chunk['index']:>2} | {len(chunk['text']):>5} chars"
    )


async def main():
    """
    Main function for running the scraper client test.
    
    Usage:
        python src/services/scraper_client.py
    """
    # Test with a sample file
    test_file = "/app/assets/Alpha.pdf"
    
    if os.path.exists(test_file):
        logger.info(f"Testing with file: {test_file}")
        await send_files_and_stream_chunks([test_file])
    else:
        logger.error(f"Test file not found: {test_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
