# Minimal Dockerfile - No requirements.txt issues
FROM python:3.9-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Working directory
WORKDIR /app

# Install dependencies one by one (no requirements.txt)
RUN pip install --no-cache-dir \
    flask==2.3.3 \
    ccxt==4.1.0 \
    requests==2.31.0 \
    pandas==2.1.0 \
    numpy==1.24.0 \
    python-telegram-bot==20.5

# Copy only main application file
COPY main.py ./
COPY config.py ./

# Create a minimal config if not exists
RUN echo "# Minimal config for Cloud Run" > config_minimal.py

# Create directories
RUN mkdir -p logs models backtest_cache reports data

# Expose port
EXPOSE $PORT

# Health check with curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Run main.py with minimal dependencies
CMD python main.py --test-local