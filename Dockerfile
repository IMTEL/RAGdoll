# Dockerfile for RAGdoll FastAPI Application
# Purpose: Defines how to build the application container image
# - Used by docker-compose for local development
# - Used directly for production deployment (Kubernetes, cloud platforms, CI/CD)
# - Creates a standalone, portable container with all dependencies
# - Optimized layer caching: dependencies installed before code copy
# Use an official Python base image

FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_NO_DEV=1 \
    PATH="/root/.local/bin/:$PATH"

# Set the working directory
WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Download uv package manager (cached layer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

FROM base AS packages
COPY uv.lock pyproject.toml ./
RUN uv sync --locked --no-dev

FROM base AS runtime
COPY --from=packages /app/.venv /app/.venv

# Install system dependencies (cached layer)
# - curl, git: for uv and general utilities    && apt-get 
# - libgl1, libglib2.0-0: required by OpenCV (cv2) used by unstructured library
# - poppler-utils: required for PDF processing (pdftotext, pdfinfo, etc.)
# - tesseract-ocr: optional OCR support for scanned PDFs
# - ffmpeg: required by Whisper for browser-recorded formats such as WebM/Opus
RUN apt-get update && \
    apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application (this layer rebuilds on code changes, but deps are cached)
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
