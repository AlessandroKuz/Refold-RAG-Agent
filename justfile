# Refold RAG Agent justfile

export PYTHONPATH := "src"

# List available commands
default:
    @just --list

# Copy template env file if not present
env:
    @if [ ! -f .env ]; then cp .env.example .env && echo "Created .env from .env.example"; else echo ".env already exists"; fi

# Pull required Ollama models
pull-models:
    ollama pull gpt-oss
    ollama pull qwen3-embedding

# alias for the `pull-models` command
models: pull-models

# Run test suite
test:
    @if [ -f .venv/bin/pytest ]; then .venv/bin/pytest; else pytest; fi

# Force rebuild Chroma vector index from resources/
index:
    @if [ -f .venv/bin/python ]; then .venv/bin/python main.py --reindex; else python main.py --reindex; fi

# Start FastAPI RAG server
run:
    @if [ -f .venv/bin/python ]; then .venv/bin/python main.py; else python main.py; fi

# Start OpenWebUI frontend container in Docker with host networking (Linux)
openwebui:
    @docker run -d \
      --network=host \
      -e PORT=3000 \
      -v open-webui:/app/backend/data \
      --name open-webui \
      --restart always \
      ghcr.io/open-webui/open-webui:main || true
    @echo "OpenWebUI running at http://localhost:3000"

# Stop and remove OpenWebUI container
openwebui-stop:
    -docker stop open-webui
    -docker rm open-webui

# Purge OpenWebUI container and data volume to reset admin account
openwebui-reset:
    -docker stop open-webui
    -docker rm open-webui
    -docker volume rm open-webui
    @echo "OpenWebUI data purged."
    @just openwebui

# Check backend health status
health:
    curl -s http://localhost:8000/health

# View recent SQLite audit logs
logs limit="10":
    curl -s "http://localhost:8000/v1/audit/logs?limit={{limit}}"

# Clean setup and run: create .env -> pull models -> run tests -> launch OpenWebUI -> index & start server
run-everything: env models test openwebui
    @if [ -f .venv/bin/python ]; then .venv/bin/python main.py --reindex; else python main.py --reindex; fi
