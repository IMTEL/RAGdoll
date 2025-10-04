# Use an official Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Ensure the local bin is in PATH for uv
    PATH="/root/.local/bin:$PATH"

# Install git (and optionally build tools for other dependencies)
RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Download uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy and install dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync

# Copy the rest of the application
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
