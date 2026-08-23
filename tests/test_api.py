"""Integration tests for FastAPI endpoints."""

from unittest.mock import AsyncMock, MagicMock
import pytest
from fastapi.testclient import TestClient

from main import app
from refold_agent.api import get_audit_logger, get_rag_engine
from refold_agent.db import AuditLogger
from refold_agent.rag import RAGEngine


@pytest.fixture
def test_client(test_settings):
    """FastAPI TestClient with overridden dependencies."""
    audit = AuditLogger(test_settings.sqlite_db_path)
    mock_rag = MagicMock(spec=RAGEngine)
    mock_rag.vector_store = MagicMock()
    mock_rag.vector_store._collection = MagicMock()
    mock_rag.vector_store._collection.count.return_value = 45

    app.dependency_overrides[get_rag_engine] = lambda: mock_rag
    app.dependency_overrides[get_audit_logger] = lambda: audit

    with TestClient(app, raise_server_exceptions=True) as client:
        yield client, mock_rag

    app.dependency_overrides.clear()


def test_health_endpoint(test_client):
    """Test /health status endpoint."""
    client, _ = test_client
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["chroma_documents_count"] == 45
    assert "models" in data


def test_list_models_endpoint(test_client):
    """Test /v1/models endpoint for OpenWebUI."""
    client, _ = test_client
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "refold-agent"


def test_query_endpoint(test_client):
    """Test /query endpoint returning answer and sources."""
    client, mock_rag = test_client
    mock_rag.answer_query = AsyncMock(return_value={
        "query": "What is Phase 1?",
        "rephrased_query": "What is Phase 1 Foundations?",
        "answer": "Phase 1 is foundations.\n\n### Sources:\n- `phase1a.md`",
        "sources": [{"source": "phase1a.md", "score": 0.85}],
        "scores": [0.85],
        "has_context": True,
    })

    response = client.post("/query", json={"query": "What is Phase 1?"})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "What is Phase 1?"
    assert data["has_context"] is True
    assert "phase1a.md" in data["answer"]
    assert len(data["sources"]) == 1


def test_chat_completions_non_streaming(test_client):
    """Test /v1/chat/completions non-streaming response."""
    client, mock_rag = test_client
    mock_rag.answer_query = AsyncMock(return_value={
        "query": "Hello",
        "rephrased_query": "Hello",
        "answer": "Hello! I am the Refold assistant.",
        "sources": [],
        "scores": [],
        "has_context": True,
    })

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "refold-agent",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "Hello! I am the Refold assistant."


def test_chat_completions_streaming(test_client):
    """Test /v1/chat/completions streaming (SSE) response."""
    client, mock_rag = test_client

    async def mock_stream(_messages):
        yield {"type": "token", "content": "Hello "}
        yield {"type": "token", "content": "world!"}
        yield {
            "type": "token",
            "content": "\n\n### Sources:\n- `phase0a.md`",
            "metadata": {"rephrased_query": "Hello", "sources": [], "has_context": True},
        }

    mock_rag.stream_chat_completion = mock_stream

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "refold-agent",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    content = response.text
    assert "data: " in content
    assert "[DONE]" in content
