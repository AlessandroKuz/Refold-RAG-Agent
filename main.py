"""Application entry point with FastAPI lifespan, CLI argument parsing, and uvicorn runner."""

import argparse
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure src/ is on sys.path for direct script execution without editable install
SRC_PATH = Path(__file__).resolve().parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from refold_agent.api import router
from refold_agent.config import get_settings
from refold_agent.db import AuditLogger
from refold_agent.rag import RAGEngine

# CLI flags storage
FORCE_REINDEX = False


def setup_logging(log_level: str) -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan managing database connection and index initialization."""
    settings = get_settings()
    setup_logging(settings.log_level)
    logger = logging.getLogger("refold_agent")

    logger.info("Initializing Refold RAG Agent (Ollama chat=%s, embed=%s)", settings.ollama_chat_model, settings.ollama_embedding_model)

    # Initialize SQLite audit logger
    audit_logger = AuditLogger(settings.sqlite_db_path)
    app.state.audit_logger = audit_logger

    # Initialize RAG Engine
    rag_engine = RAGEngine(settings)
    app.state.rag_engine = rag_engine

    # Initialize or load Chroma vector index
    try:
        total_indexed = rag_engine.initialize_index(force_reindex=FORCE_REINDEX)
        logger.info("Vector index ready with %d documents.", total_indexed)
    except Exception as e:
        logger.error("Failed to initialize vector index: %s", e)

    yield

    logger.info("Shutting down Refold RAG Agent.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Refold RAG Agent API",
        version="0.1.0",
        description="Local RAG agent API for Refold language learning methodology documents.",
        lifespan=lifespan,
    )

    # Enable CORS for OpenWebUI frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Refold RAG Agent Server")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force rebuild the ChromaDB vector index from resources/",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host binding (defaults to HOST env var or 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port binding (defaults to PORT env var or 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload for development",
    )
    return parser.parse_known_args()[0]


def main() -> None:
    """CLI execution entrypoint."""
    global FORCE_REINDEX
    args = parse_args()

    if args.reindex:
        FORCE_REINDEX = True

    settings = get_settings()
    host = args.host or settings.host
    port = args.port or settings.port

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
