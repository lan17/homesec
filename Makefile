SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: help up down docker-build docker-push run db test coverage typecheck lint lock-check check db-migrate db-migration publish ui-% fake-camera

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
	@echo "    make run           Run HomeSec locally (requires Postgres)"
	@echo "    make db            Start just Postgres"
	@echo "    make test          Run tests with coverage"
	@echo "    make coverage      Run tests and generate HTML coverage report"
	@echo "    make typecheck     Run mypy"
	@echo "    make lint          Run ruff linter"
	@echo "    make lock-check    Verify uv.lock is up to date"
	@echo "    make check         Run lint + typecheck + test + ui-check"
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
run:
	@echo "Running database migrations..."
	@$(UV_RUN) alembic -c alembic.ini upgrade head
	$(UV_RUN) python -m homesec.cli run --config $(HOMESEC_CONFIG) --log_level $(HOMESEC_LOG_LEVEL)

db:
	docker compose up -d postgres

test:
	$(UV_RUN) pytest tests/homesec/ -v --cov=homesec --cov-report=term-missing

coverage:
	$(UV_RUN) pytest tests/homesec/ -v --cov=homesec --cov-report=html --cov-report=xml
	@echo "Coverage report: htmlcov/index.html"

typecheck:
	$(UV_RUN) mypy --package homesec --strict

lint:
	$(UV_RUN) ruff check src tests
	$(UV_RUN) ruff format --check src tests

lint-fix:
	$(UV_RUN) ruff check --fix src tests
	$(UV_RUN) ruff format src tests

lock-check:
	uv lock --check

check: lock-check lint typecheck test ui-check

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
