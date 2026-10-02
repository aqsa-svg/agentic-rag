# Hugging Face Spaces serves on 7860 and runs the container as a non-root user.
FROM python:3.11-slim

# Only the online extra: no torch, no PyMuPDF. Keeps the image small and makes the
# import-boundary test's "no offline deps on the request path" rule true at runtime too.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ARAG_ENV=production \
    ARAG_LOG_LEVEL=INFO \
    ARAG_ENGINE=null

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . && \
    useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:7860/health')"

CMD ["uvicorn", "arag.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
