SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: help homesec test typecheck check
.PHONY: db-up db-down db-logs db-migrate db-migrate-homesec
.PHONY: pypi-clean pypi-build pypi-check pypi-publish
.PHONY: docker-build docker-push docker-run docker-up docker-down docker-logs

help:
	@echo "Targets:"
	@echo "  make homesec    Run HomeSec pipeline (RTSP motion detection + YOLO + Dropbox)"
	@echo "  make test       Run tests"
	@echo "  make typecheck  Run mypy strict type checking"
	@echo "  make check      Run both test + typecheck"
	@echo "  make db-up      Start local Postgres (telemetry)"
	@echo "  make db-down    Stop local Postgres (telemetry)"
	@echo "  make db-logs    Tail Postgres logs (telemetry)"
	@echo "  make db-migrate Run alembic migrations (telemetry + clip state/events)"
	@echo "  make db-migrate-homesec Run alembic migrations (clip state/events)"
	@echo "  make pypi-clean    Remove local build artifacts (dist/, build/)"
	@echo "  make pypi-build    Build sdist/wheel for PyPI"
	@echo "  make pypi-check    Validate dist artifacts with twine"
	@echo "  make pypi-publish  Build, check, and upload to PyPI"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-push   Push Docker image to DockerHub"
	@echo "  make docker-run    Run Docker container locally (standalone)"
	@echo "  make docker-up     Start HomeSec + Postgres via docker-compose"
	@echo "  make docker-down   Stop docker-compose services"
	@echo "  make docker-logs   Tail docker-compose logs"
	@echo "  DB_DSN=$${DB_DSN:-}"

HOMESEC_CONFIG ?= config/production.yaml
HOMESEC_LOG_LEVEL ?= INFO

test:
	uv run pytest tests/homesec/ -v

typecheck:
	uv run mypy --package homesec --strict

check: typecheck test

homesec:
	@set -a && source .env 2>/dev/null; set +a; \
	uv run python -m homesec.cli run --config $(HOMESEC_CONFIG) --log_level $(HOMESEC_LOG_LEVEL)

db-up:
	docker compose up -d postgres

db-down:
	docker compose down postgres

db-logs:
	docker compose logs -f postgres

db-migrate:
	uv run --with alembic --with sqlalchemy --with asyncpg alembic -c alembic.ini upgrade head

db-migrate-homesec:
	$(MAKE) db-migrate

pypi-clean:
	rm -rf dist build

pypi-build: pypi-clean check
	uv run --with build python -m build

pypi-check:
	uv run --with twine python -m twine check dist/*

pypi-publish: pypi-build pypi-check
	uv run --with twine python -m twine upload dist/*

# Docker
DOCKER_IMAGE ?= homesec
DOCKER_TAG ?= latest
DOCKERHUB_USER ?= $(shell echo $${DOCKERHUB_USER:-})

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

docker-run:
	docker run --rm -it \
		-v $$(pwd)/config/example.yaml:/config/config.yaml \
		-v $$(pwd)/.env:/config/.env \
		-v $$(pwd)/recordings:/data/recordings \
		-v $$(pwd)/storage:/data/storage \
		-v $$(pwd)/yolo_cache:/app/yolo_cache \
		-p 8080:8080 \
		$(DOCKER_IMAGE):$(DOCKER_TAG)

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f
