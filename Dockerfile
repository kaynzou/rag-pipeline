FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -e ".[dev,local]"
RUN pip install --no-cache-dir "datasets>=2.14.0"

COPY src/ ./src/
COPY data/ ./data/
COPY ingest_msmarco_xi.py ./ingest_msmarco_xi.py
COPY app.py ./app.py

RUN mkdir -p data/index

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
