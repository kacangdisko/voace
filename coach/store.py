import json
import sqlite3
from pathlib import Path

from .config import DB_PATH


def _db(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY, created_at TEXT, report TEXT)")
    return con


def save(report: dict, path: Path = DB_PATH) -> int:
    with _db(path) as con:
        cur = con.execute("INSERT INTO sessions (created_at, report) VALUES (?, ?)",
                          (report["created_at"], json.dumps(report)))
        return cur.lastrowid


def load(session_id: int, path: Path = DB_PATH) -> dict | None:
    with _db(path) as con:
        row = con.execute("SELECT report FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return json.loads(row[0]) if row else None
