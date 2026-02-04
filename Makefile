SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: help up down docker-build docker-push run db test coverage typecheck lint check db-migrate db-migration publish ha-lint ha-test

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
	@echo "    make check         Run lint + typecheck + test"
	@echo "    make ha-lint       Run ruff on HA integration code"
	@echo "    make ha-test       Run HA integration tests"
	@echo ""
	@echo "  Database:"
	@echo "    make db-migrate    Run migrations"
	@echo "    make db-migration m=\"description\"  Generate new migration"
	@echo ""
	@echo "  Release:"
	@echo "    make publish       Build and upload to PyPI"

# Config
HOMESEC_CONFIG ?= config/config.yaml
HOMESEC_LOG_LEVEL ?= INFO
DOCKER_IMAGE ?= homesec
DOCKER_TAG ?= latest
DOCKERHUB_USER ?= $(shell echo $${DOCKERHUB_USER:-})
HA_INTEGRATION_DIR := homeassistant/integration
HA_INTEGRATION_SRC := $(HA_INTEGRATION_DIR)/custom_components
HA_INTEGRATION_TESTS := $(HA_INTEGRATION_DIR)/tests

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
	@uv run alembic -c alembic.ini upgrade head
	uv run python -m homesec.cli run --config $(HOMESEC_CONFIG) --log_level $(HOMESEC_LOG_LEVEL)

db:
	docker compose up -d postgres

test:
	uv run pytest tests/homesec/ -v --cov=homesec --cov-report=term-missing

coverage:
	uv run pytest tests/homesec/ -v --cov=homesec --cov-report=html --cov-report=xml
	@echo "Coverage report: htmlcov/index.html"

typecheck:
	uv run mypy --package homesec --strict

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
	make -C homeassistant shellcheck
	make ha-lint

ha-lint:
	uv run ruff check $(HA_INTEGRATION_SRC) $(HA_INTEGRATION_TESTS)
	uv run ruff format --check $(HA_INTEGRATION_SRC) $(HA_INTEGRATION_TESTS)

ha-test:
	uv run pytest $(HA_INTEGRATION_TESTS) -q

lint-fix:
	uv run ruff check --fix src tests
	uv run ruff format src tests

check: lint typecheck test ha-lint ha-test

# Database
db-migrate:
	uv run --with alembic --with sqlalchemy --with asyncpg --with python-dotenv alembic -c alembic.ini upgrade head

db-migration:
	@if [ -z "$(m)" ]; then \
		echo "Error: message required. Run: make db-migration m=\"your description\""; \
		exit 1; \
	fi
	uv run --with alembic --with sqlalchemy --with asyncpg --with python-dotenv alembic -c alembic.ini revision --autogenerate -m "$(m)"

# Release
publish: check
	rm -rf dist build
	uv run --with build python -m build
	uv run --with twine python -m twine check dist/*
	uv run --with twine python -m twine upload dist/*
