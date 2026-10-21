# Sift service image (CPU by default; use a CUDA base image + torch wheel for GPU).
# Not built in the development environment for this project (disk constraints) —
# treat as a reference deployment recipe.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/models SIFT_HOST=0.0.0.0 SIFT_PORT=8002 USE_TF=0

WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY sift ./sift
COPY web ./web
RUN pip install --no-cache-dir . \
 && python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Datasets + indexes are built on first start into the mounted volume.
VOLUME ["/app/data"]
EXPOSE 8002
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8002/api/health')"
CMD ["sh", "-c", "python -m sift.datasets download && python -m sift.server"]
