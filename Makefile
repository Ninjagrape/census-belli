# General WAR — developer tasks.
# Recipe lines are tab-indented; `make` rejects spaces there.

.DEFAULT_GOAL := help
SHELL := /bin/sh

COMPOSE ?= docker compose
PYTHON  ?= python
DB_SERVICE := postgres

# Load .env so DATABASE_URL and the POSTGRES_* vars reach both compose and
# the Python tooling. Missing .env is not fatal: targets that need it say so.
ifneq (,$(wildcard .env))
include .env
export
endif

.PHONY: help install lint typecheck test db-up db-down db-reset db-schema db-shell db-wait \n        db-migrate db-migrate-sql db-revision

help:  ## Show available targets
	grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

install:  ## Install the package and dev dependencies
	$(PYTHON) -m pip install -e ".[dev]"

lint:  ## Run ruff
	$(PYTHON) -m ruff check pipeline/ scripts/ tests/

typecheck:  ## Run mypy
	$(PYTHON) -m mypy pipeline/

test:  ## Run the test suite
	$(PYTHON) -m pytest tests/ -v

db-up:  ## Start Postgres and wait until it accepts connections
	$(COMPOSE) up -d $(DB_SERVICE)
	$(MAKE) db-wait

db-wait:  ## Block until the container reports healthy
	printf 'waiting for postgres'
	for i in $$(seq 1 40); do \
		if [ "$$($(COMPOSE) ps -q $(DB_SERVICE) | xargs -r docker inspect -f '{{.State.Health.Status}}' 2>/dev/null)" = "healthy" ]; then \
			printf ' ready\n'; exit 0; \
		fi; \
		printf '.'; sleep 1; \
	done; \
	printf '\ntimed out after 40s; try: $(COMPOSE) logs $(DB_SERVICE)\n'; exit 1

db-down:  ## Stop Postgres, keeping data
	$(COMPOSE) down

db-schema:  ## Apply config/schema.sql to the running database
	$(PYTHON) -m pipeline.db --apply-schema

db-reset:  ## Destroy the volume and rebuild from schema (DESTRUCTIVE)
	@printf 'This deletes the general_war volume and all crawled data. Ctrl-C to abort.\n'
	@sleep 3
	$(COMPOSE) down -v
	$(MAKE) db-up
	$(MAKE) db-schema

db-migrate:  ## Apply Alembic migrations up to head
	PYTHONIOENCODING=utf-8 $(PYTHON) -m alembic upgrade head

db-migrate-sql:  ## Print the migration SQL without applying it
	PYTHONIOENCODING=utf-8 $(PYTHON) -m alembic upgrade head --sql

db-revision:  ## Create a migration: make db-revision m="add x"
	PYTHONIOENCODING=utf-8 $(PYTHON) -m alembic revision -m "$(m)"

db-shell:  ## Open psql against the running database
	$(COMPOSE) exec $(DB_SERVICE) psql -U $${POSTGRES_USER:-general_war} -d $${POSTGRES_DB:-general_war}
