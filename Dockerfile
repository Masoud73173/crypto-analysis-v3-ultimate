# Ultra-Simple Dockerfile for Cloud Run - No Dependencies Issues
FROM python:3.9-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Install only essential system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install only core dependencies first
RUN pip install --no-cache-dir \
    flask \
    ccxt \
    requests \
    pandas \
    numpy

# Copy application files
COPY *.py ./

# Create directories
RUN mkdir -p logs models backtest_cache reports data

# Expose port
EXPOSE $PORT

# Simple health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Run command
CMD python main.py --test-local