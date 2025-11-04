# Asynchronous Document Upload

## Overview

The document upload system now processes files asynchronously in the background, keeping the server responsive even during large file uploads or processing of many documents.

## How It Works

### 1. Upload Returns Immediately

When you upload a document via `POST /upload/agent`, the endpoint:

- Validates authentication and agent existence
- Saves the file temporarily
- **Returns immediately** with a `task_id`
- Processes the document in the background

### 2. Background Processing

The background task:

- Extracts text from the document (PDF, DOCX, TXT, MD, etc.)
- Chunks the text for optimal RAG performance
- Computes embeddings for each chunk
- Stores everything in the database
- Cleans up temporary files
- Updates progress status

### 3. Progress Tracking

You can check the status of your upload using the `task_id`:

```bash
GET /upload/status/{task_id}
```

Or view all progress logs:

```bash
GET /api/progress
```

## API Usage

### Upload a Document

```bash
POST /upload/agent?agent_id={agent_id}
Content-Type: multipart/form-data

file: <your_file>
```

**Response:**

```json
{
  "message": "Document uploaded successfully, processing in background",
  "filename": "document.pdf",
  "agent_id": "agent-123",
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "size_bytes": 1024000
}
```

### Check Upload Status

```bash
GET /upload/status/550e8400-e29b-41d4-a716-446655440000
```

**Response (Processing):**

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "document.pdf",
  "agent_id": "agent-123",
  "status": "processing",
  "message": "Processing document.pdf...",
  "document_id": null,
  "started_at": "2025-11-01T12:34:56.789Z",
  "completed_at": null
}
```

**Response (Complete):**

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "document.pdf",
  "agent_id": "agent-123",
  "status": "complete",
  "message": "Successfully processed document.pdf",
  "document_id": "doc-uuid-here",
  "started_at": "2025-11-01T12:34:56.789Z",
  "completed_at": "2025-11-01T12:35:23.456Z"
}
```

## Status Values

| Status       | Description                                                   |
| ------------ | ------------------------------------------------------------- |
| `queued`     | Upload received, waiting to start processing                  |
| `processing` | Currently extracting text, chunking, and computing embeddings |
| `complete`   | Successfully processed and stored in database                 |
| `failed`     | Processing failed (check logs for details)                    |
| `error`      | An error occurred during processing                           |

## Benefits

1. **Server Responsiveness**: Server can handle multiple requests while processing large documents
2. **Better UX**: Users get immediate feedback and can check progress
3. **Scalability**: Can handle multiple concurrent uploads without blocking
4. **Reliability**: Errors in one upload don't affect others

## Implementation Details

### FastAPI BackgroundTasks

The implementation uses FastAPI's built-in `BackgroundTasks`, which:

- Runs tasks in a thread pool
- Doesn't block the main request/response cycle
- Automatically handles task lifecycle
- Is simple and doesn't require external dependencies

### Thread Safety

The scraper and embedding services are thread-safe for read operations. The database DAOs handle their own connection pooling and thread safety.

### Configuration

Default chunking parameters in `scraper_service`:

- `chunk_size`: 1000 characters
- `overlap`: 100 characters

These can be adjusted in `context_upload.py` if needed.

## Future Enhancements

Possible improvements:

1. Add WebSocket support for real-time progress updates
2. Implement batch upload endpoint
3. Add retry logic for failed uploads
4. Store progress in database instead of in-memory
5. Add upload queue visualization in admin panel
