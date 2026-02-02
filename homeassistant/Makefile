SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

ADDON_DIR := addon/homesec
SHELLCHECK ?= shellcheck

S6_SCRIPTS := $(shell find $(ADDON_DIR)/rootfs/etc/s6-overlay/s6-rc.d -type f \( -name run -o -name up \))

.PHONY: help shellcheck addon-build

help:
	@echo "Targets:"
	@echo ""
	@echo "  make shellcheck    Run shellcheck on add-on s6 scripts"
	@echo "  make addon-build   Build the HomeSec add-on Docker image"

shellcheck:
	@if ! command -v $(SHELLCHECK) >/dev/null; then \
		echo "shellcheck not found. Install shellcheck to run this target."; \
		exit 1; \
	fi
	@if [ -z "$(S6_SCRIPTS)" ]; then \
		echo "No s6 scripts found under $(ADDON_DIR)/rootfs."; \
		exit 0; \
	fi
	$(SHELLCHECK) --shell=bash $(S6_SCRIPTS)

addon-build:
	docker build -t homesec-addon $(ADDON_DIR)
