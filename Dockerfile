FROM python:3.11-slim

# Unbuffered output means container logs appear immediately, which matters when
# debugging a demo through `docker compose logs -f`.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_TEMP_DIR=/tmp/gradio \
    EMNIST_MODEL_PATH=/model_weights       
    

WORKDIR /app

# Dependencies are copied and installed before the source so that editing an app
# does not invalidate the (slow) pip layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as an unprivileged user rather than root.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /tmp/gradio \
    && chown -R appuser:appuser /app /tmp/gradio
USER appuser

RUN chmod 744 /app/apps/*.py

EXPOSE 7860


# `curl` is not present in the slim image, so probe with the stdlib instead.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/healthz')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
