"""
Voice Pipeline SQLite Database.

Lightweight, zero-config persistence layer for conversation history,
session tracking, and configuration storage.
"""

import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / ".voice_pipeline" / "voice_pipeline.db"


@dataclass
class Conversation:
    """Conversation record."""

    id: str
    session_id: str
    user_text: str
    llm_response: str
    stt_language: str
    tts_voice: str
    audio_path: Optional[str] = None
    audio_duration: Optional[float] = None
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class Session:
    """Session record."""

    id: str
    created_at: str = ""
    ended_at: Optional[str] = None
    turn_count: int = 0
    total_duration: float = 0.0


@dataclass
class ConfigEntry:
    """Configuration entry."""

    key: str
    value: str


class Database:
    """
    SQLite database manager for Voice Pipeline.

    Thread-safe, auto-creates tables on first use.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    @contextmanager
    def _connect(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_tables(self) -> None:
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    ended_at TEXT,
                    turn_count INTEGER DEFAULT 0,
                    total_duration REAL DEFAULT 0.0
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_text TEXT NOT NULL,
                    llm_response TEXT NOT NULL,
                    stt_language TEXT NOT NULL,
                    tts_voice TEXT NOT NULL,
                    audio_path TEXT,
                    audio_duration REAL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS audio_files (
                    id TEXT PRIMARY KEY,
                    path TEXT UNIQUE NOT NULL,
                    duration REAL,
                    size_bytes INTEGER,
                    status TEXT DEFAULT 'pending',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);
                CREATE INDEX IF NOT EXISTS idx_conv_created ON conversations(created_at);
                CREATE INDEX IF NOT EXISTS idx_session_created ON sessions(created_at);
            """)

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """Convert sqlite3.Row to dict."""
        return dict(row)

    # --- Sessions ---

    def create_session(self) -> Session:
        """Create a new session."""
        session = Session(
            id=str(uuid.uuid4()),
            created_at=datetime.now().isoformat(),
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, created_at) VALUES (?, ?)",
                (session.id, session.created_at),
            )
        logger.info(f"Session created: {session.id}")
        return session

    def end_session(self, session_id: str) -> None:
        """End a session."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET ended_at = ? WHERE id = ?",
                (datetime.now().isoformat(), session_id),
            )

    def get_session(self, session_id: str) -> Optional[dict]:
        """Get session by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            return self._row_to_dict(row) if row else None

    def list_sessions(self, limit: int = 50) -> list[dict]:
        """List recent sessions."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def update_session_stats(
        self, session_id: str, turn_count: int, total_duration: float
    ) -> None:
        """Update session statistics."""
        with self._connect() as conn:
            conn.execute(
                """UPDATE sessions 
                   SET turn_count = ?, total_duration = ? 
                   WHERE id = ?""",
                (turn_count, total_duration, session_id),
            )

    # --- Conversations ---

    def add_conversation(self, conv: Conversation) -> str:
        """Add a conversation record."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO conversations 
                   (id, session_id, user_text, llm_response, stt_language, 
                    tts_voice, audio_path, audio_duration, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    conv.id,
                    conv.session_id,
                    conv.user_text,
                    conv.llm_response,
                    conv.stt_language,
                    conv.tts_voice,
                    conv.audio_path,
                    conv.audio_duration,
                    conv.created_at,
                ),
            )
        logger.debug(f"Conversation saved: {conv.id}")
        return conv.id

    def get_conversation(self, conv_id: str) -> Optional[dict]:
        """Get conversation by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conv_id,)
            ).fetchone()
            return self._row_to_dict(row) if row else None

    def list_conversations(
        self, session_id: Optional[str] = None, limit: int = 50
    ) -> list[dict]:
        """List recent conversations, optionally filtered by session."""
        with self._connect() as conn:
            if session_id:
                rows = conn.execute(
                    """SELECT * FROM conversations 
                       WHERE session_id = ? 
                       ORDER BY created_at DESC LIMIT ?""",
                    (session_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM conversations ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def search_conversations(self, query: str, limit: int = 20) -> list[dict]:
        """Search conversations by user text or LLM response."""
        search_term = f"%{query}%"
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM conversations 
                   WHERE user_text LIKE ? OR llm_response LIKE ?
                   ORDER BY created_at DESC LIMIT ?""",
                (search_term, search_term, limit),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM conversations WHERE id = ?", (conv_id,)
            )
            return cursor.rowcount > 0

    def clear_conversations(self) -> int:
        """Clear all conversations."""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM conversations")
            return cursor.rowcount

    # --- Config ---

    def get_config(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a config value."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM config WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else default

    def set_config(self, key: str, value: str) -> None:
        """Set a config value (upsert)."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO config (key, value) VALUES (?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = ?""",
                (key, value, value),
            )

    def get_all_config(self) -> dict[str, str]:
        """Get all config entries."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM config").fetchall()
            return {r["key"]: r["value"] for r in rows}

    def delete_config(self, key: str) -> bool:
        """Delete a config entry."""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM config WHERE key = ?", (key,))
            return cursor.rowcount > 0

    # --- Audio Files ---

    def add_audio_file(
        self, path: str, duration: Optional[float] = None, size_bytes: Optional[int] = None
    ) -> str:
        """Add an audio file record."""
        file_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO audio_files (id, path, duration, size_bytes, status, created_at)
                   VALUES (?, ?, ?, ?, 'processed', ?)""",
                (file_id, path, duration, size_bytes, datetime.now().isoformat()),
            )
        return file_id

    def list_audio_files(self, limit: int = 50) -> list[dict]:
        """List recent audio files."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM audio_files ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    # --- Stats ---

    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        with self._connect() as conn:
            total_conversations = conn.execute(
                "SELECT COUNT(*) FROM conversations"
            ).fetchone()[0]
            total_sessions = conn.execute(
                "SELECT COUNT(*) FROM sessions"
            ).fetchone()[0]
            total_audio_duration = conn.execute(
                "SELECT COALESCE(SUM(audio_duration), 0) FROM conversations"
            ).fetchone()[0]
            total_audio_files = conn.execute(
                "SELECT COUNT(*) FROM audio_files"
            ).fetchone()[0]

            return {
                "total_conversations": total_conversations,
                "total_sessions": total_sessions,
                "total_audio_duration_seconds": round(total_audio_duration, 2),
                "total_audio_files": total_audio_files,
            }

    def export_conversations(self) -> list[dict]:
        """Export all conversations as list of dicts."""
        return self.list_conversations(limit=10000)
