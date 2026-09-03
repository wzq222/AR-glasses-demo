import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


class Database:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def migrate(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT NOT NULL UNIQUE,
                  display_name TEXT NOT NULL,
                  password_hash TEXT NOT NULL,
                  role TEXT NOT NULL CHECK(role IN ('admin','inspector','reviewer')),
                  active INTEGER NOT NULL DEFAULT 1,
                  created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sop_templates (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  code TEXT NOT NULL,
                  version INTEGER NOT NULL,
                  title TEXT NOT NULL,
                  description TEXT NOT NULL DEFAULT '',
                  steps_json TEXT NOT NULL,
                  active INTEGER NOT NULL DEFAULT 1,
                  created_by INTEGER NOT NULL REFERENCES users(id),
                  created_at TEXT NOT NULL,
                  UNIQUE(code, version)
                );
                CREATE TABLE IF NOT EXISTS assignments (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  template_id INTEGER NOT NULL REFERENCES sop_templates(id),
                  assignee_id INTEGER NOT NULL REFERENCES users(id),
                  asset_code TEXT NOT NULL,
                  status TEXT NOT NULL CHECK(status IN ('pending','in_progress','completed','cancelled')),
                  due_at TEXT,
                  created_by INTEGER NOT NULL REFERENCES users(id),
                  created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                  id TEXT PRIMARY KEY,
                  assignment_id INTEGER NOT NULL REFERENCES assignments(id),
                  operator_id INTEGER NOT NULL REFERENCES users(id),
                  status TEXT NOT NULL CHECK(status IN ('in_progress','submitted','reviewed','rejected')),
                  device_json TEXT NOT NULL DEFAULT '{}',
                  started_at TEXT NOT NULL,
                  submitted_at TEXT,
                  reviewed_by INTEGER REFERENCES users(id),
                  reviewed_at TEXT,
                  review_note TEXT
                );
                CREATE TABLE IF NOT EXISTS step_results (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  run_id TEXT NOT NULL REFERENCES runs(id),
                  step_key TEXT NOT NULL,
                  idempotency_key TEXT NOT NULL UNIQUE,
                  status TEXT NOT NULL CHECK(status IN ('succeeded','uncertain','failed','skipped')),
                  value_json TEXT NOT NULL DEFAULT '{}',
                  confidence REAL,
                  requires_human_review INTEGER NOT NULL,
                  human_decision TEXT,
                  analyzer_version TEXT NOT NULL,
                  error_code TEXT,
                  captured_at TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  UNIQUE(run_id, step_key)
                );
                CREATE TABLE IF NOT EXISTS evidence (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  step_result_id INTEGER NOT NULL REFERENCES step_results(id),
                  storage_name TEXT NOT NULL UNIQUE,
                  original_name TEXT NOT NULL,
                  media_type TEXT NOT NULL,
                  sha256 TEXT NOT NULL,
                  size_bytes INTEGER NOT NULL,
                  created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  actor_id INTEGER REFERENCES users(id),
                  action TEXT NOT NULL,
                  entity_type TEXT NOT NULL,
                  entity_id TEXT NOT NULL,
                  detail_json TEXT NOT NULL DEFAULT '{}',
                  created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_assignments_assignee ON assignments(assignee_id, status);
                CREATE INDEX IF NOT EXISTS idx_runs_assignment ON runs(assignment_id);
                CREATE INDEX IF NOT EXISTS idx_steps_run ON step_results(run_id);
                """
            )

    @staticmethod
    def row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row else None

    @staticmethod
    def json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
