SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: help up down docker-build docker-push dev-setup run db test coverage typecheck lint lock-check check rust-check rust-build db-migrate db-migration publish ui-% fake-camera

help:
	@echo "Targets:"
	@echo ""
	@echo "  Docker:"
	@echo "    make up            Start HomeSec + Postgres"
	@echo "    make down          Stop all services"
	@echo "    make docker-build  Build Docker image"
	@echo "    make docker-push   Push to DockerHub"
	@echo ""
	@echo "  Local dev:"
	@echo "    make dev-setup     Check tools, install locked dependencies, build helper + UI"
	@echo "    make run           Run HomeSec locally (requires Postgres)"
	@echo "    make db            Start just Postgres"
	@echo "    make test          Run tests with coverage"
	@echo "    make coverage      Run tests and generate HTML coverage report"
	@echo "    make typecheck     Run mypy"
	@echo "    make lint          Run ruff linter"
	@echo "    make lock-check    Verify uv.lock is up to date"
	@echo "    make rust-build    Build the Rust WebRTC helper"
	@echo "    make rust-check    Run Rust formatting, Clippy, and tests"
	@echo "    make check         Run Rust + Python + UI checks"
	@echo "    make fake-camera   Start a mock ONVIF + RTSP camera (requires ffmpeg, mediamtx)"
	@echo ""
	@echo "  Database:"
	@echo "    make db-migrate    Run migrations"
	@echo "    make db-migration m=\"description\"  Generate new migration"
	@echo ""
	@echo "  Release:"
	@echo "    make publish       Build and upload to PyPI"
	@echo ""
	@echo "  UI proxy:"
	@echo "    make ui-<target>   Run make target in ui/ (example: make ui-api-generate)"

# Config
HOMESEC_CONFIG ?= config/config.yaml
HOMESEC_LOG_LEVEL ?= INFO
DOCKER_IMAGE ?= homesec
DOCKER_TAG ?= latest
DOCKERHUB_USER ?= $(shell echo $${DOCKERHUB_USER:-})
UV_RUN ?= uv run --locked
CARGO ?= cargo
PYTHON ?= python3
FFMPEG_JOBS ?=
PNPM ?= pnpm
WEBRTC_MANIFEST := native/webrtc/Cargo.toml
NATIVE_BUILD := $(PYTHON) native/webrtc/build_native.py $(if $(FFMPEG_JOBS),--jobs $(FFMPEG_JOBS)) --
WEBRTC_RELEASE_DIR := $(if $(CARGO_TARGET_DIR),$(CARGO_TARGET_DIR),$(CURDIR)/native/webrtc/target)/release

# Docker
up:
	docker compose up -d --build

down:
	docker compose down

docker-build:
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-push: docker-build
	@if [ -z "$(DOCKERHUB_USER)" ]; then \
		echo "Error: DOCKERHUB_USER not set. Run: export DOCKERHUB_USER=yourusername"; \
		exit 1; \
	fi
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKERHUB_USER)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKERHUB_USER)/$(DOCKER_IMAGE):latest
	docker push $(DOCKERHUB_USER)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	docker push $(DOCKERHUB_USER)/$(DOCKER_IMAGE):latest

# Local dev
dev-setup:
	@for tool in uv node ffmpeg ffprobe clang pkg-config make cmake; do \
		if ! command -v "$$tool" >/dev/null 2>&1; then \
			echo "Missing $$tool. See docs/webrtc-preview.md for developer prerequisites."; \
			exit 1; \
		fi; \
	done
	@$(PNPM) --version >/dev/null || { \
		echo "pnpm is unavailable. Install the version in ui/package.json; see docs/webrtc-preview.md."; \
		exit 1; \
	}
	@$(CARGO) --version >/dev/null || { \
		echo "Cargo or the pinned Rust toolchain is unavailable. Install Rust via rustup; see docs/webrtc-preview.md."; \
		exit 1; \
	}
	@$(CC) --version >/dev/null || { \
		echo "A C compiler/linker is required. See docs/webrtc-preview.md for Linux/macOS prerequisites."; \
		exit 1; \
	}
	@$(CXX) --version >/dev/null || { \
		echo "A C++ compiler/linker is required to build bundled OpenCV. See docs/webrtc-preview.md."; \
		exit 1; \
	}
	@$(PYTHON) --version >/dev/null || { echo "Python 3 is required to build bundled native libraries."; exit 1; }
	uv sync --locked
	$(MAKE) rust-build
	$(MAKE) ui-install
	$(MAKE) ui-build
	@echo "Developer setup complete. WebRTC helper: $(WEBRTC_RELEASE_DIR)/homesec-webrtc"

run:
	@echo "Running database migrations..."
	@$(UV_RUN) alembic -c alembic.ini upgrade head
	PATH="$(WEBRTC_RELEASE_DIR):$$PATH" $(UV_RUN) python -m homesec.cli run --config $(HOMESEC_CONFIG) --log_level $(HOMESEC_LOG_LEVEL)

db:
	docker compose up -d postgres

test:
	$(UV_RUN) pytest tests/homesec/ -v --cov=homesec --cov-report=term-missing

coverage:
	$(UV_RUN) pytest tests/homesec/ -v --cov=homesec --cov-report=html --cov-report=xml
	@echo "Coverage report: htmlcov/index.html"

typecheck:
	$(UV_RUN) mypy --package homesec --strict
	$(UV_RUN) mypy native/webrtc/build_native.py --strict

lint:
	$(UV_RUN) ruff check src tests native/webrtc/build_native.py
	$(UV_RUN) ruff format --check src tests native/webrtc/build_native.py

lint-fix:
	$(UV_RUN) ruff check --fix src tests native/webrtc/build_native.py
	$(UV_RUN) ruff format src tests native/webrtc/build_native.py

lock-check:
	uv lock --check

rust-build:
	$(NATIVE_BUILD) $(CARGO) build --manifest-path $(WEBRTC_MANIFEST) --release --locked

rust-check:
	$(CARGO) fmt --manifest-path $(WEBRTC_MANIFEST) --all -- --check
	$(NATIVE_BUILD) $(CARGO) clippy --manifest-path $(WEBRTC_MANIFEST) --all-targets --locked -- -D warnings
	$(NATIVE_BUILD) $(CARGO) test --manifest-path $(WEBRTC_MANIFEST) --locked

check: lock-check rust-check lint typecheck test ui-check

fake-camera:
	@echo "Starting mock ONVIF server on port 8000..."
	@python3 dev/fake-camera/mock_onvif.py &
	@echo "Starting RTSP server on port 8099..."
	@./mediamtx dev/fake-camera/mediamtx.yml &
	@sleep 2
	@echo "Streaming media/sample.mp4 to rtsp://localhost:8099/live..."
	@ffmpeg -re -stream_loop -1 -i media/sample.mp4 -c copy -f rtsp rtsp://admin:admin123@localhost:8099/live

# Database
db-migrate:
	$(UV_RUN) --with alembic --with sqlalchemy --with asyncpg --with python-dotenv alembic -c alembic.ini upgrade head

db-migration:
	@if [ -z "$(m)" ]; then \
		echo "Error: message required. Run: make db-migration m=\"your description\""; \
		exit 1; \
	fi
	$(UV_RUN) --with alembic --with sqlalchemy --with asyncpg --with python-dotenv alembic -c alembic.ini revision --autogenerate -m "$(m)"

# Release
publish: check
	rm -rf dist build
	$(UV_RUN) --with build python -m build
	$(UV_RUN) --with twine python -m twine check dist/*
	$(UV_RUN) --with twine python -m twine upload dist/*

# Proxy any ui-* target to the UI Makefile (e.g., ui-api-generate -> make -C ui api-generate).
ui-%:
	@$(MAKE) -C ui $*
