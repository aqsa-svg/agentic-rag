# IRDAI-only public service. Base deps + pymupdf only: NO torch, NO sentence-transformers,
# NO onnxruntime. Retrieval is BM25 (pure Python) over the two public IRDAI PDFs, built at
# startup; generation is the hosted Gemini API. The four insurer wordings are never in this
# image - only the public regulatory documents are baked in.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ARAG_ENV=production \
    ARAG_LOG_LEVEL=INFO \
    ARAG_ENGINE=agent \
    ARAG_DEPLOY_DIR=/app/data/deploy \
    ARAG_RAW_DIR=/app/data/raw \
    ARAG_TOP_K=5 \
    PORT=8080

WORKDIR /app

# Base package (fastapi/uvicorn/pydantic/structlog/httpx) + pymupdf for the startup corpus build.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . "pymupdf>=1.24" && \
    useradd -m -u 1000 appuser

# Bake ONLY the public IRDAI sources, their page census, and the IRDAI-only manifest.
# The four insurer PDFs are deliberately NOT copied - they are copyrighted and local-only.
COPY data/deploy/irdai_sources.jsonl data/deploy/probe_report.json /app/data/deploy/
COPY data/raw/irdai-master-circular-2024.pdf data/raw/irdai-annexure-2024.pdf /app/data/raw/
RUN chown -R appuser /app
USER appuser

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import os,urllib.request;urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8080\")}/health')"

# $PORT is honoured (Render/Railway inject it; Fly maps internal_port to it). Shell form so the
# variable expands.
CMD uvicorn arag.api.app:app --host 0.0.0.0 --port ${PORT:-8080}
