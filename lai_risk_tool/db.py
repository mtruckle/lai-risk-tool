"""
Data models and SQLite persistence for positions.
"""
import sqlite3
import json
from pathlib import Path
from datetime import date, datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

DB_PATH = Path(__file__).parent / "positions.db"


def init_db():
    """Create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bbg_ticker TEXT NOT NULL,
            instrument_type TEXT NOT NULL,  -- 'OPTION' or 'STOCK'
            underlying TEXT NOT NULL,
            option_root TEXT,   -- for options: symbol root used for yfinance (e.g. SOXS, SOXS1)
            expiry TEXT,        -- YYYY-MM-DD for options
            strike REAL,
            option_type TEXT,   -- 'C' or 'P'
            multiplier REAL NOT NULL,  -- shares per contract, typically 100, or 5 for post-split
            quantity REAL NOT NULL,    -- signed: +long / -short  (contracts for options, shares for stock)
            entry_price REAL NOT NULL,
            entry_date TEXT NOT NULL,
            exit_price REAL,
            exit_date TEXT,
            status TEXT NOT NULL,   -- 'OPEN' or 'CLOSED'
            notes TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def insert_position(pos: dict) -> int:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    cols = ",".join(pos.keys())
    qs = ",".join("?" * len(pos))
    c.execute(f"INSERT INTO positions ({cols}) VALUES ({qs})", list(pos.values()))
    new_id = c.lastrowid
    conn.commit()
    conn.close()
    return new_id


def update_position(pos_id: int, updates: dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    assignments = ",".join(f"{k}=?" for k in updates.keys())
    c.execute(f"UPDATE positions SET {assignments} WHERE id=?", list(updates.values()) + [pos_id])
    conn.commit()
    conn.close()


def delete_position(pos_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM positions WHERE id=?", (pos_id,))
    conn.commit()
    conn.close()


def get_positions(status: Optional[str] = None):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    if status:
        c.execute("SELECT * FROM positions WHERE status=? ORDER BY id", (status,))
    else:
        c.execute("SELECT * FROM positions ORDER BY id")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_setting(key: str, default: str = None) -> str:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT value FROM settings WHERE key=?", (key,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else default


def set_setting(key: str, value: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()
