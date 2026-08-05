# AutoTuneX — convenience targets.
#
# The whole stack (API + UI, API Bridge, MySQL) runs via docker-compose from
# this directory. Run `make help` for the list of targets. For the full
# reference and optional IBM extras, see README.md ("Docker Deployment").

# Prefer the Docker Compose v2 plugin; fall back to the legacy binary.
COMPOSE ?= docker compose
UX_DIR  := ux

.DEFAULT_GOAL := help

.PHONY: help env up down build logs ps restart clean check lint format ux-install

help: ## Show this help
	@echo "AutoTuneX — available targets:"
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# ---- Stack lifecycle (docker-compose) ---------------------------------------

env: ## Create .env from .env.example if it does not exist (then edit secrets)
	@if [ -f .env ]; then \
		echo ".env already exists — leaving it untouched."; \
	else \
		cp .env.example .env && \
		echo "Created .env from .env.example — edit the passwords and SESSION_SECRET before 'make up'."; \
	fi

up: ## Build and start the full stack (mysql, app, bridge) in the background
	$(COMPOSE) up -d --build

build: ## Build all service images without starting them
	$(COMPOSE) build

down: ## Stop the stack (keeps the DB + data volumes)
	$(COMPOSE) down

restart: ## Restart the app (API + UI) service
	$(COMPOSE) restart app

logs: ## Follow the API/UI (app) logs
	$(COMPOSE) logs -f app

ps: ## Show the status of the stack's containers
	$(COMPOSE) ps

clean: ## Stop the stack AND drop the DB + data volumes (destructive)
	$(COMPOSE) down -v

# ---- Frontend checks (ux/) --------------------------------------------------

ux-install: ## Install frontend dependencies (ux/)
	cd $(UX_DIR) && npm install

check: ## Run Svelte/TypeScript type checking (ux/)
	cd $(UX_DIR) && npm run check

lint: ## Run Prettier check + ESLint (ux/)
	cd $(UX_DIR) && npm run lint

format: ## Auto-format the frontend with Prettier (ux/)
	cd $(UX_DIR) && npm run format
