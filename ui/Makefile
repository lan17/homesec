SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

PNPM ?= pnpm
VITE_API_PROXY_TARGET ?=
VITE_API_BASE_URL ?=
VITE_DEV_HOST ?= 0.0.0.0
VITE_DEV_PORT ?=

.PHONY: help install run-local dev api-generate api-check lint test build check

help:
	@echo "UI targets:"
	@echo "  make run-local   Start Vite dev server"
	@echo "  make install     Install dependencies"
	@echo "  make api-generate  Regenerate typed API artifacts"
	@echo "  make api-check   Verify generated API artifacts are up to date"
	@echo "  make lint        Run eslint"
	@echo "  make test        Run vitest"
	@echo "  make build       Run typecheck + Vite build"
	@echo "  make check       Run full UI gate (api-check + lint + test + build)"
	@echo ""
	@echo "Optional env vars:"
	@echo "  VITE_API_PROXY_TARGET=http://127.0.0.1:8081"
	@echo "  VITE_API_BASE_URL=http://127.0.0.1:8081"
	@echo "  VITE_DEV_HOST=0.0.0.0"
	@echo "  VITE_DEV_PORT=5173"

install:
	$(PNPM) install

run-local:
	VITE_API_PROXY_TARGET="$(VITE_API_PROXY_TARGET)" \
	VITE_API_BASE_URL="$(VITE_API_BASE_URL)" \
	$(PNPM) exec vite --host "$(VITE_DEV_HOST)" $(if $(VITE_DEV_PORT),--port "$(VITE_DEV_PORT)",)

dev: run-local

api-generate:
	$(PNPM) api:generate

api-check:
	$(PNPM) api:check

lint:
	$(PNPM) lint

test:
	$(PNPM) test

build:
	$(PNPM) build

check:
	$(PNPM) check
