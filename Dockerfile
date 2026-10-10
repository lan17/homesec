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
#          -p 8081:8081 homesec

# =============================================================================
# Stage 1: WebRTC Helper Builder
# =============================================================================
FROM rust:1.99.0-slim-bookworm AS webrtc-builder

# Build pinned FFmpeg and OpenCV libraries privately and link them into the helper.
# Both native binding generators need libclang during the build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    python3 \
    cmake \
    clang \
    libclang-dev \
    pkg-config \
    nasm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Makefile ./
COPY rust-toolchain.toml ./
COPY LICENSE ./
COPY native/webrtc/ ./native/webrtc/
RUN make rust-build \
    && install -D -m 0644 /tmp/homesec-native-0/ffmpeg-*/COPYING.LGPLv2.1 \
        /app/ffmpeg-license/COPYING.LGPLv2.1 \
    && install -D -m 0644 /tmp/homesec-native-0/ffmpeg-*/COPYING.GPLv3 \
        /app/ffmpeg-license/COPYING.GPLv3 \
    && install -D -m 0644 /tmp/homesec-native-0/x264-*/COPYING \
        /app/x264-license/COPYING \
    && install -D -m 0644 /tmp/homesec-native-0/opus-*/COPYING \
        /app/opus-license/COPYING \
    && install -D -m 0644 /tmp/homesec-native-0/opencv-*/LICENSE \
        /app/opencv-license/LICENSE \
    && install -D -m 0644 /tmp/homesec-native-0/opencv-*/ZLIB-LICENSE \
        /app/opencv-license/ZLIB-LICENSE \
    && install -D -m 0644 /tmp/homesec-native-0/ffmpeg-8.1.3.tar.xz \
        /app/native-source/ffmpeg-8.1.3.tar.xz \
    && install -D -m 0644 /tmp/homesec-native-0/x264-*.tar.gz \
        /app/native-source/x264.tar.gz \
    && install -D -m 0644 /tmp/homesec-native-0/opus-*.tar.gz \
        /app/native-source/opus.tar.gz \
    && install -D -m 0644 /tmp/homesec-native-0/opencv-*.tar.gz \
        /app/native-source/opencv.tar.gz \
    && mkdir -p .cargo \
    && cargo vendor --manifest-path native/webrtc/Cargo.toml --locked --versioned-dirs \
        native/webrtc/vendor/registry > .cargo/config.toml \
    && tar --exclude=native/webrtc/target -czf /app/native-source/homesec-native-source.tar.gz \
        Makefile rust-toolchain.toml LICENSE .cargo/config.toml native/webrtc

# =============================================================================
# Stage 2: Python Builder
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
# Stage 3: UI Builder
# =============================================================================
FROM node:22-bookworm-slim AS ui-builder

WORKDIR /app/ui

# Use pinned package manager from ui/package.json via corepack.
RUN corepack enable

# Copy lockfile first for better build caching.
COPY ui/package.json ui/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

# Copy UI sources and build static assets.
COPY ui/ ./
RUN pnpm build

# =============================================================================
# Stage 4: Runtime
# =============================================================================
FROM python:3.14-slim-bookworm AS runtime

# Install runtime dependencies
# - ffmpeg: CLI recording/transcoding and compatibility motion
# - libgl1: required by OpenCV
# - libglib2.0-0: required by OpenCV
# - postgresql-client-16: pg_dump/pg_restore version compatible with docker-compose postgres:16
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    && install -d -m 0755 /usr/share/postgresql-common/pgdg \
    && curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc \
        | gpg --dearmor -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.gpg \
    && echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.gpg] https://apt.postgresql.org/pub/repos/apt bookworm-pgdg main" \
        > /etc/apt/sources.list.d/pgdg.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libstdc++6 \
    postgresql-client-16 \
    && apt-get purge -y --auto-remove curl gnupg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash homesec

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/alembic /app/alembic
COPY --from=builder /app/alembic.ini /app/alembic.ini
COPY --from=ui-builder /app/ui/dist /app/ui/dist
COPY --from=webrtc-builder /app/native/webrtc/target/release/homesec-webrtc /usr/local/bin/homesec-webrtc
COPY --from=webrtc-builder /app/ffmpeg-license/ /usr/share/licenses/homesec-ffmpeg/
COPY native/webrtc/FFMPEG-NOTICE.md /usr/share/licenses/homesec-ffmpeg/NOTICE.md
COPY --from=webrtc-builder /app/x264-license/ /usr/share/licenses/homesec-x264/
COPY native/webrtc/X264-NOTICE.md /usr/share/licenses/homesec-x264/NOTICE.md
COPY --from=webrtc-builder /app/opus-license/ /usr/share/licenses/homesec-opus/
COPY native/webrtc/OPUS-NOTICE.md /usr/share/licenses/homesec-opus/NOTICE.md
COPY --from=webrtc-builder /app/native-source/ /usr/share/homesec-native/source/
COPY --from=webrtc-builder /app/opencv-license/ /usr/share/licenses/homesec-opencv/
COPY native/webrtc/OPENCV-NOTICE.md /usr/share/licenses/homesec-opencv/NOTICE.md
COPY native/webrtc/OPENCV-BINDINGS-LICENSE /usr/share/licenses/homesec-opencv/OPENCV-BINDINGS-LICENSE
# Check loading and reject shared OpenCV or libav dependencies.
RUN homesec-webrtc --help > /dev/null \
    && dependencies="$(ldd /usr/local/bin/homesec-webrtc)" \
    && ! printf '%s\n' "$dependencies" \
        | grep -E 'lib(opencv_[^[:space:]]*|avcodec|avformat|avfilter|avutil|avdevice|swscale|swresample|x264|opus)\.so'

# Copy entrypoint script
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

# Set up environment
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV HOMESEC_SERVER_UI_DIST_DIR=/app/ui/dist

# Create directories for volume mounts and make entrypoint executable
RUN chmod +x /app/docker-entrypoint.sh \
    && mkdir -p /config /data/recordings /data/storage /app/yolo_cache \
    && chown -R homesec:homesec /config /data /app

# Switch to non-root user
USER homesec

# Health check endpoint
EXPOSE 8081
# Direct WebRTC media; signaling uses the existing HTTP port.
EXPOSE 8189-8199/udp
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8081/health')" || exit 1

# Entrypoint runs migrations then starts app
# Config and env are expected to be mounted at /config/
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["run", "--config", "/config/config.yaml", "--log_level", "INFO"]
