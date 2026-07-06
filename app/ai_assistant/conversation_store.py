"""Conversation history stores: recent chat turns per session.

Two interchangeable backends behind the same tiny interface (`history` / `add`):
`InMemoryConversationStore` (process-only, reset on restart) and
`SQLiteConversationStore` (survives restarts). `CONVERSATION_STORE` at the bottom is the
single instance the flow uses; the backend is chosen by `STORE_BACKEND`.

This is separate from the flow-state persistence in `persistence.py`.
"""

import sqlite3
import threading

from app.ai_assistant.db import DB_PATH, STORE_BACKEND


class InMemoryConversationStore:
    """Process-memory history, keyed by session_id. Reset on restart."""

    def __init__(self) -> None:
        self._by_session: dict[str, list[dict]] = {}

    def history(self, session_id: str) -> list[dict]:
        # Return a copy so callers can't mutate the stored list by accident.
        return list(self._by_session.get(session_id, []))

    def add(self, session_id: str, role: str, content: str) -> None:
        self._by_session.setdefault(session_id, []).append({'role': role, 'content': content})


class SQLiteConversationStore:
    """SQLite-backed history so it survives restarts. One row per message."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()  # serialize writes; SQLite file is shared
        with self._connect() as conn:
            conn.execute(
                'CREATE TABLE IF NOT EXISTS conversations ('
                '  id INTEGER PRIMARY KEY AUTOINCREMENT,'
                '  session_id TEXT NOT NULL,'
                '  role TEXT NOT NULL,'
                '  content TEXT NOT NULL,'
                "  created_at TEXT NOT NULL DEFAULT (datetime('now'))"
                ')'
            )
            conn.execute('CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id, id)')

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def history(self, session_id: str) -> list[dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                'SELECT role, content FROM conversations WHERE session_id = ? ORDER BY id',
                (session_id,),
            ).fetchall()
        return [{'role': role, 'content': content} for role, content in rows]

    def add(self, session_id: str, role: str, content: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                'INSERT INTO conversations (session_id, role, content) VALUES (?, ?, ?)',
                (session_id, role, content),
            )


# The single instance the flow uses. 'memory' keeps the in-memory store; anything else
# (default) uses SQLite.
if STORE_BACKEND == 'memory':
    CONVERSATION_STORE: InMemoryConversationStore | SQLiteConversationStore = InMemoryConversationStore()
else:
    CONVERSATION_STORE = SQLiteConversationStore(DB_PATH)
