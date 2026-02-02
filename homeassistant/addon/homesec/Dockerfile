ARG BUILD_FROM=ghcr.io/hassio-addons/base:15.0.8
FROM ${BUILD_FROM}

ARG HOMESEC_VERSION=1.2.2

ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        ffmpeg \
        curl \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && if apt-cache show postgresql-16 >/dev/null 2>&1; then \
        apt-get install -y --no-install-recommends postgresql-16 postgresql-client-16; \
    else \
        apt-get install -y --no-install-recommends postgresql postgresql-client; \
    fi \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir "homesec==${HOMESEC_VERSION}"

COPY rootfs/ /

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/health || exit 1
