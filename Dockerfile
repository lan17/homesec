# HomeSec Dockerfile
# Multi-stage build for minimal image size
#
# Build: docker build -t homesec .
# Run:   docker run \
#          -v ./config.yaml:/config/config.yaml \
#          -v ./.env:/config/.env \
#          -v ./recordings:/data/recordings \
#          -v ./storage:/data/storage \
#          -v ./yolo_cache:/app/yolo_cache \
#          -p 8080:8080 homesec

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.14-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock* LICENSE README.md ./

# Install dependencies into a virtual environment
RUN uv venv /app/.venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

RUN uv sync --frozen --no-dev --no-install-project

# Copy source code
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini ./

# Install the project (ensure homesec is in site-packages)
RUN uv pip install --no-deps .

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.14-slim-bookworm AS runtime

# Install runtime dependencies
# - ffmpeg: required for RTSP source video processing
# - libgl1: required by OpenCV
# - libglib2.0-0: required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash homesec

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/alembic /app/alembic
COPY --from=builder /app/alembic.ini /app/alembic.ini

# Copy entrypoint script
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

# Set up environment
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Create directories for volume mounts and make entrypoint executable
RUN chmod +x /app/docker-entrypoint.sh \
    && mkdir -p /config /data/recordings /data/storage /app/yolo_cache \
    && chown -R homesec:homesec /config /data /app

# Switch to non-root user
USER homesec

# Health check endpoint
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Entrypoint runs migrations then starts app
# Config and env are expected to be mounted at /config/
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["run", "--config", "/config/config.yaml", "--log_level", "INFO"]
