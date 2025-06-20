# Simple Dockerfile for Cloud Run - Crypto Analysis Bot V3.0
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib binary (faster than compiling)
RUN pip install --no-cache-dir TA-Lib-Precompiled || pip install --no-cache-dir TA-Lib

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY config.py .
COPY analyzer.py .
COPY news_analyzer.py .
COPY ml_predictor.py .
COPY pattern_recognition.py .
COPY backtesting_engine.py .
COPY signal_formatter.py .
COPY main.py .

# Create necessary directories
RUN mkdir -p logs models backtest_cache reports data

# Expose port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Simple startup command
CMD python main.py --test-local