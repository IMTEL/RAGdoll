# Dockerfile for RAGdoll FastAPI Application
# Purpose: Defines how to build the application container image
# - Used by docker-compose for local development
# - Used directly for production deployment (Kubernetes, cloud platforms, CI/CD)
# - Creates a standalone, portable container with all dependencies
# - Optimized layer caching: dependencies installed before code copy

# Use an official Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Ensure the local bin is in PATH for uv
    PATH="/root/.local/bin:$PATH"

# Install system dependencies (cached layer)
# - curl, git: for uv and general utilities
# - libgl1, libglib2.0-0: required by OpenCV (cv2) used by unstructured library
# - poppler-utils: required for PDF processing (pdftotext, pdfinfo, etc.)
# - tesseract-ocr: optional OCR support for scanned PDFs
RUN apt-get update && \
    apt-get install -y \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Download uv package manager (cached layer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy ONLY dependency files first (this layer is cached unless dependencies change)
COPY pyproject.toml uv.lock ./

# Install dependencies globally to system Python (cached layer - only rebuilds when pyproject.toml or uv.lock changes)
# No venv needed - the container itself provides isolation
RUN uv pip install --system -r pyproject.toml

# Copy the rest of the application (this layer rebuilds on code changes, but deps are cached)
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run FastAPI with Uvicorn (no 'uv run' needed - packages are in system Python)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
