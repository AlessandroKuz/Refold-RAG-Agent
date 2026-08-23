"""SQLite database audit logging service for chat queries and completions."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditLogger:
    """Audit logger persisting queries, retrieved chunks, and responses in SQLite."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a connection with standard timeouts and row factory."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize SQLite tables and indexes."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    session_id TEXT,
                    endpoint TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    rephrased_query TEXT,
                    retrieved_sources TEXT,
                    answer TEXT,
                    model TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_logs(session_id)"
            )
            conn.commit()

    def log(
        self,
        endpoint: str,
        prompt: str,
        answer: str,
        model: str,
        session_id: str | None = None,
        rephrased_query: str | None = None,
        retrieved_sources: list[dict[str, Any]] | None = None,
    ) -> int:
        """Insert a single audit log entry and return its ID."""
        now = datetime.now(timezone.utc).isoformat()
        sources_json = json.dumps(retrieved_sources or [])

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO audit_logs (
                    timestamp, session_id, endpoint, prompt,
                    rephrased_query, retrieved_sources, answer, model
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    session_id,
                    endpoint,
                    prompt,
                    rephrased_query,
                    sources_json,
                    answer,
                    model,
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def get_recent_logs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Retrieve the most recent audit log entries."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, timestamp, session_id, endpoint, prompt,
                       rephrased_query, retrieved_sources, answer, model
                FROM audit_logs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            return [
                {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "session_id": row["session_id"],
                    "endpoint": row["endpoint"],
                    "prompt": row["prompt"],
                    "rephrased_query": row["rephrased_query"],
                    "retrieved_sources": json.loads(row["retrieved_sources"]),
                    "answer": row["answer"],
                    "model": row["model"],
                }
                for row in rows
            ]
