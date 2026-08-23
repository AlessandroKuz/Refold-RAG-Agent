"""Tests for SQLite audit logging service."""

from refold_agent.db import AuditLogger


def test_audit_logger_insert_and_retrieve(test_settings):
    """Test logging and reading queries from SQLite."""
    logger = AuditLogger(test_settings.sqlite_db_path)

    log_id = logger.log(
        endpoint="/query",
        session_id="test-session-123",
        prompt="What is Phase 1?",
        rephrased_query="What is Phase 1 Foundations in Refold?",
        retrieved_sources=[{"source": "phase1a.md", "score": 0.85}],
        answer="Phase 1 is about building foundational vocabulary.",
        model="gpt-oss",
    )

    assert log_id > 0

    logs = logger.get_recent_logs(limit=10)
    assert len(logs) == 1
    entry = logs[0]
    assert entry["id"] == log_id
    assert entry["session_id"] == "test-session-123"
    assert entry["endpoint"] == "/query"
    assert entry["prompt"] == "What is Phase 1?"
    assert entry["rephrased_query"] == "What is Phase 1 Foundations in Refold?"
    assert entry["model"] == "gpt-oss"
    assert len(entry["retrieved_sources"]) == 1
    assert entry["retrieved_sources"][0]["source"] == "phase1a.md"
