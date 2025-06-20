# Final Dockerfile - با همه dependencies لازم
FROM python:3.9-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Working directory
WORKDIR /app

# Install ALL necessary dependencies (کامل!)
RUN pip install --no-cache-dir \
    flask==2.3.3 \
    flask-cors==4.0.0 \
    ccxt==4.1.0 \
    requests==2.31.0 \
    pandas==2.1.0 \
    numpy==1.24.0 \
    python-telegram-bot==20.5 \
    schedule==1.2.0 \
    python-dotenv==1.0.0 \
    pytz==2023.3

# Copy ALL Python files
COPY *.py ./

# Create necessary directories
RUN mkdir -p logs models backtest_cache reports data

# Install curl for health check
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Expose port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Debug startup command
CMD echo "Starting Bot V3.0..." && \
    echo "PORT=$PORT" && \
    echo "Python version:" && python --version && \
    echo "Starting Flask app..." && \
    python main.py --test-local