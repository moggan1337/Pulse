# Pulse - Distributed Tracing with ML Anomaly Detection
# Dockerfile

FROM python:3.11-slim

LABEL maintainer="Pulse Team <team@pulse.dev>"
LABEL description="Distributed tracing with ML-powered anomaly detection"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p /app/data

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV SERVICE_NAME=pulse
ENV STORAGE_BACKEND=memory

# Expose ports
EXPOSE 8080 4317 4318

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health').raise_for_status()"

# Run the application
CMD ["python", "-m", "src.server"]
