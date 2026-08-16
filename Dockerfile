# Dockerfile
#
# Production image for the async crypto paper-trading bot.
# Lightweight, non-root, single-stage build on python:3.11-slim.

FROM python:3.11-slim

# --- Runtime environment ---------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# --- OS-level dependencies ---------------------------------------------------
# Only what's needed to build/run the Python deps above; apt cache is
# purged in the same layer to keep the final image lean.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# --- Python dependencies -----------------------------------------------------
# Copied and installed before the rest of the source so this layer is
# cached across rebuilds when only application code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Application source -------------------------------------------------------
COPY . .

# --- Non-root user ------------------------------------------------------------
# Run as an unprivileged user rather than root inside the container.
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser \
    && chown -R appuser:appuser /app

USER appuser

# --- Entrypoint -----------------------------------------------------------
CMD ["python", "main.py"]
