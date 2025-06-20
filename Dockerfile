# Complete Dockerfile - Copy all necessary files
FROM python:3.9-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Working directory
WORKDIR /app

# Install basic dependencies
RUN pip install --no-cache-dir \
    flask==2.3.3 \
    ccxt==4.1.0 \
    requests==2.31.0 \
    pandas==2.1.0 \
    numpy==1.24.0

# Copy ALL Python files (to avoid missing dependencies)
COPY *.py ./

# Create necessary directories
RUN mkdir -p logs models backtest_cache reports data

# Create a minimal requirements file (fallback)
RUN echo "flask==2.3.3" > requirements.txt

# Install curl for health check
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Expose port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Debug startup command (shows what's happening)
CMD echo "Starting Bot V3.0..." && \
    echo "PORT=$PORT" && \
    echo "Files in /app:" && \
    ls -la && \
    echo "Starting Python..." && \
    python main.py --test-local