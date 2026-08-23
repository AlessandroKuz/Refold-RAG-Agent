"""Configuration management using pydantic-settings for 12-factor cloud readiness."""

from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host binding")
    port: int = Field(default=8000, description="Server port binding")
    log_level: str = Field(default="INFO", description="Logging level")

    # Ollama Configuration
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for Ollama service",
    )
    ollama_chat_model: str = Field(
        default="gpt-oss",
        description="Ollama model for chat generation and query rephrasing",
    )
    ollama_embedding_model: str = Field(
        default="qwen3-embedding",
        description="Ollama model for document and query embeddings",
    )

    # OpenWebUI Integration
    openwebui_model_name: str = Field(
        default="refold-agent",
        description="Model name exposed to OpenWebUI discovery",
    )

    # Storage Paths
    resources_dir: Path = Field(
        default=Path("resources"),
        description="Directory containing markdown source documents",
    )
    data_dir: Path = Field(
        default=Path("data"),
        description="Root directory for persisted data (Docker volume mount target)",
    )
    chroma_dir: Path = Field(
        default=Path("data/chroma"),
        description="Persistent ChromaDB vector index directory",
    )
    sqlite_db_path: Path = Field(
        default=Path("data/chat.db"),
        description="SQLite database file for chat audit logging",
    )

    # RAG Hyperparameters
    top_k: int = Field(
        default=8,
        description="Maximum number of context chunks to retrieve",
    )
    similarity_threshold: float = Field(
        default=0.5,
        description="Minimum cosine similarity score (0.0 - 1.0) for retrieved chunks",
    )
    chunk_size: int = Field(
        default=1000,
        description="Max character size per split chunk",
    )
    chunk_overlap: int = Field(
        default=200,
        description="Character overlap between consecutive chunks",
    )
    hybrid_pool_size: int = Field(
        default=10,
        description="Candidate chunks retrieved per branch (BM25 & Chroma) before reranking",
    )
    rerank_top_k: int = Field(
        default=6,
        description="Top chunks retained after FlashRank cross-encoder reranking",
    )
    flashrank_model: str = Field(
        default="ms-marco-TinyBERT-L-2-v2",
        description="FlashRank cross-encoder model name",
    )
    rerank_score_threshold: float = Field(
        default=0.5,
        description="Minimum score threshold for reranked chunks",
    )

    def ensure_directories(self) -> None:
        """Ensure runtime directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    settings = Settings()
    settings.ensure_directories()
    return settings
