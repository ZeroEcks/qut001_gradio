FROM python:3.11-slim

# Unbuffered output means container logs appear immediately, which matters when
# debugging a demo through `docker compose logs -f`.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_TEMP_DIR=/tmp/gradio \
    MODEL_WEIGHTS_DIR=/model_weights       

WORKDIR /app

# Copy the large model weights first so we can cache it, since this won't change much
COPY model_weights /

# Install pytorch (with a different index so it needs a separate file)
COPY requirements-torch.txt .
RUN pip install --no-cache-dir -r requirements-torch.txt

# Install the rest of pypi dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./
COPY apps/ ./apps
COPY qut001/ ./qut001

# Run as an unprivileged user rather than root.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /tmp/gradio \
    && chown -R appuser:appuser /app /tmp/gradio \
    && chmod 744 -R /app
USER appuser

EXPOSE 7860

# `curl` is not present in the slim image, so probe with the stdlib instead.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/healthz')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
