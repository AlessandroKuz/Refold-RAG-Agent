"""Pytest fixtures and configuration."""

import sys
import tempfile
from pathlib import Path

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest.mock import MagicMock
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from refold_agent.config import Settings
from refold_agent.db import AuditLogger


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmp_path:
        yield Path(tmp_path)


@pytest.fixture
def test_settings(temp_dir):
    """Provide isolated Settings instance with temporary paths."""
    return Settings(
        data_dir=temp_dir / "data",
        chroma_dir=temp_dir / "data" / "chroma",
        sqlite_db_path=temp_dir / "data" / "chat.db",
        resources_dir=temp_dir / "resources",
        similarity_threshold=0.5,
        top_k=4,
    )


@pytest.fixture
def sample_markdown_dir(temp_dir):
    """Create a sample directory with markdown documents for testing."""
    resources = temp_dir / "resources"
    resources.mkdir(parents=True, exist_ok=True)

    doc1 = resources / "phase0-test.md"
    doc1.write_text(
        "# Phase 0 — Immersion\n\n"
        "Immersion is the key to natural acquisition.\n\n"
        "## Subphase 0A\n\n"
        "Start with passive and active listening.",
        encoding="utf-8",
    )

    doc2 = resources / "phase1-test.md"
    doc2.write_text(
        "# Phase 1 — Foundations\n\n"
        "Build the habit.\n\n"
        "## Core Vocabulary\n\n"
        "Learn the top 1000 words with SRS flashcards.",
        encoding="utf-8",
    )

    return resources


@pytest.fixture
def audit_logger(test_settings):
    """Provide an initialized AuditLogger instance."""
    return AuditLogger(test_settings.sqlite_db_path)
