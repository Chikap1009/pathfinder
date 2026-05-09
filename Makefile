.PHONY: help up down logs ps restart \
        api-dev web-dev dev \
        lint format typecheck test \
        clean

help:
	@echo "PathFinder Make targets:"
	@echo "  up         — docker compose up -d (qdrant, neo4j, redis)"
	@echo "  down       — docker compose down"
	@echo "  logs       — tail compose logs"
	@echo "  ps         — list compose services"
	@echo "  restart    — down + up"
	@echo "  api-dev    — uvicorn --reload on :8000"
	@echo "  web-dev    — pnpm next dev on :3000"
	@echo "  dev        — api-dev + web-dev concurrently"
	@echo "  lint       — ruff + biome"
	@echo "  format     — ruff format + biome format"
	@echo "  typecheck  — mypy + tsc --noEmit"
	@echo "  test       — pytest + vitest"
	@echo "  clean      — remove caches"

up:
	docker compose up -d qdrant neo4j redis

down:
	docker compose down

logs:
	docker compose logs -f --tail=200

ps:
	docker compose ps

restart: down up

api-dev:
	uv --directory apps/api run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

web-dev:
	pnpm --filter web dev

dev:
	@command -v concurrently >/dev/null 2>&1 || pnpm add -w -D concurrently
	pnpm exec concurrently -n api,web -c blue,green "$(MAKE) api-dev" "$(MAKE) web-dev"

lint:
	uv --directory apps/api run ruff check .
	pnpm --filter web lint

format:
	uv --directory apps/api run ruff format .
	pnpm --filter web format

typecheck:
	uv --directory apps/api run mypy app
	pnpm --filter web typecheck

test:
	uv --directory apps/api run pytest
	pnpm --filter web test

clean:
	find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".mypy_cache" -o -name ".ruff_cache" -o -name ".next" -o -name ".turbo" \) -prune -exec rm -rf {} +
	rm -rf apps/web/node_modules node_modules
