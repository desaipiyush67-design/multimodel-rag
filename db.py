"""Minimal local SQLite logging module."""
import os
import sqlite3
from datetime import datetime

DB_PATH = os.environ.get("CHAT_DB_PATH", "chat_logs.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            source TEXT,
            question TEXT,
            answer TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def log_chat(question: str, answer: str, source: str = "PDF Document"):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO chat_logs (ts, source, question, answer) VALUES (?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), source, question, answer),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[db.log_chat] {e}")
