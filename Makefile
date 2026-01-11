SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: help homesec test typecheck check
.PHONY: db-up db-down db-logs db-migrate db-migrate-homesec
.PHONY: pypi-clean pypi-build pypi-check pypi-publish

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
	docker compose -f docker-compose.postgres.yml up -d

db-down:
	docker compose -f docker-compose.postgres.yml down

db-logs:
	docker compose -f docker-compose.postgres.yml logs -f

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
