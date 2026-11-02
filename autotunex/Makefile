.DEFAULT_GOAL := help
.PHONY: help install install-training audit dev dev-api-bridge docker-api-bridge test test-cov test-subprojects test-fm-tune lint format typecheck check migration migrate migrations-check downgrade publish-image clean

PYTHON ?= python
PORT ?= 8000
API_BRIDGE_PORT ?= 8001
API_BRIDGE_IMAGE ?= autotunex-api-bridge:local
RUNTIME_IMAGE ?= autotunex-runtime:local
INSTALL_POSTGRES ?= 0
GH_USER ?=
GH_TOKEN ?=
DOCKER ?= docker
# The published CUDA image. A tag must keep pointing at one image: bump the suffix
# for a new build, the prefix with the CUDA / torch pins.
CUDA_IMAGE_VERSION ?= cu130-torch2.11-1
CUDA_IMAGE_REF ?= quay.io/granitebuild/autotunex:$(CUDA_IMAGE_VERSION)
PUBLISH_OVERWRITE ?= false
TAG_STATUS := ../steps/scripts/registry-tag-status.sh

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with dev dependencies
	uv pip install -e ".[dev]" || $(PYTHON) -m pip install -e ".[dev]"
	uv pip install -e "./src/fm-tune" || $(PYTHON) -m pip install -e "./src/fm-tune"

install-training:  ## Also install the heavy fm-tune training stack ([full],mlx) for the local runner
	uv pip install -e "./src/fm-tune[full,mlx]" || $(PYTHON) -m pip install -e "./src/fm-tune[full,mlx]"

audit:  ## Fail if any installed dependency has a known CVE
	uvx pip-audit

dev:  ## Run the API locally with autoreload
	uvicorn autotunex.main:app --reload --port $(PORT)

dev-api-bridge:  ## Run the api-bridge logging server on port 8001 (override API_BRIDGE_PORT=...; needs: pip install ./src/api-bridge)
	cd src/api-bridge && API_BRIDGE_SERVER_PORT=$(API_BRIDGE_PORT) PYTHONPATH=src $(PYTHON) -m api_bridge.server

docker-api-bridge:  ## Build the api-bridge image (override API_BRIDGE_IMAGE=...; add INSTALL_POSTGRES=1 for psycopg)
	docker build -t $(API_BRIDGE_IMAGE) --build-arg INSTALL_POSTGRES=$(INSTALL_POSTGRES) src/api-bridge

test:  ## Run the test suite
	pytest

test-cov:  ## Run tests with a coverage report
	pytest --cov=autotunex --cov-report=term-missing --cov-report=xml

test-subprojects:  ## Install and run the standalone subproject test suites (api-bridge)
	uv pip install -e "./src/api-bridge[dev]" || $(PYTHON) -m pip install -e "./src/api-bridge[dev]"
	cd src/api-bridge && pytest

test-fm-tune:  ## Run the vendored fm-tune suite (needs: pip install ./src/fm-tune[core]); skips slow/gpu/full tests
	cd src/fm-tune && $(PYTHON) -m pytest -m "not slow and not gpu and not full"

lint:  ## Check lint and formatting
	ruff check .
	ruff format --check .
	ruff check src/api-bridge
	ruff format --check src/api-bridge
	ruff check src/fm-tune
	ruff format --check src/fm-tune

format:  ## Apply lint fixes and formatting
	ruff check --fix .
	ruff format .
	ruff check --fix src/api-bridge
	ruff format src/api-bridge
	ruff check --fix src/fm-tune
	ruff format src/fm-tune

typecheck:  ## Run mypy in strict mode
	mypy

check: lint typecheck test test-subprojects  ## Everything CI runs

migration:  ## Autogenerate a migration: make migration m="add trials table"
	@test -n "$(m)" || (echo 'Usage: make migration m="description"' && exit 1)
	alembic revision --autogenerate -m "$(m)"

migrate:  ## Apply migrations up to head
	alembic upgrade head

migrations-check:  ## Fail if db/tables/ has drifted from the migrations
	alembic check

downgrade:  ## Roll back one migration
	alembic downgrade -1

publish-image:  ## Build the CUDA image and push it to CUDA_IMAGE_REF; refuses an existing tag (PUBLISH_OVERWRITE=true to replace)
	@if [ "$(PUBLISH_OVERWRITE)" = "true" ]; then \
		echo "PUBLISH_OVERWRITE=true: not checking whether $(CUDA_IMAGE_REF) exists."; \
	else \
		status=$$($(TAG_STATUS) "$(CUDA_IMAGE_REF)"); \
		case "$$status" in \
			absent) echo "$(CUDA_IMAGE_REF) is a new tag." ;; \
			present) echo "ERROR: $(CUDA_IMAGE_REF) already exists; bump CUDA_IMAGE_VERSION, or pass PUBLISH_OVERWRITE=true to replace it on purpose." >&2; exit 1 ;; \
			*) echo "ERROR: could not tell whether $(CUDA_IMAGE_REF) exists. Log in first (podman login quay.io, or export REGISTRY_USER and REGISTRY_PASSWORD)." >&2; exit 1 ;; \
		esac; \
	fi
	$(DOCKER) build --platform linux/amd64 --build-arg DEVICE=cuda -t $(CUDA_IMAGE_REF) .
	$(DOCKER) push $(CUDA_IMAGE_REF)

clean:  ## Remove caches and build artifacts
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +

