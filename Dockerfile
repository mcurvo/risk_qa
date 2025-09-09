# Slim Python, works on Linux/AMD64 & Apple Silicon (via buildx)
FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirement spec first (leverage Docker layer cache)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the app
COPY app /app/app
COPY scripts /app/scripts
COPY data /app/data
COPY .env /app/.env

# Expose port
EXPOSE 8000

# Healthcheck hits /health
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD curl -fsS http://localhost:8000/health || exit 1

# Default command: run API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
