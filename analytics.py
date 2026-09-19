"""
Flora Carbon AI - Real session analytics log (stdlib sqlite3, zero new dependency).

Every completed analysis is logged here so dashboard numbers (history trend,
global counters, leaderboard) are genuinely computed from what this instance
has actually processed - never hardcoded placeholders. The log is trimmed to
hardware.CONFIG["analytics_history_limit"] rows so it stays small on
low-RAM/low-disk tiers.
"""

import os
import sqlite3
import time
import logging
from typing import Dict, Any, List

import hardware

logger = logging.getLogger("FloraAnalytics")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "flora_analytics.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    filename TEXT,
    tree_count INTEGER,
    canopy_m2 REAL,
    cover_pct REAL,
    co2e_tons REAL,
    inference_time_ms REAL,
    source TEXT
)
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_SCHEMA)
    return conn


def log_analysis(
    filename: str,
    tree_count: int,
    canopy_m2: float,
    cover_pct: float,
    co2e_tons: float,
    inference_time_ms: float,
    source: str = "analyze"
) -> None:
    """Records one real analysis run. Fast local SQLite write (a few ms) - called
    synchronously right after a successful /api/analyze so it never affects the
    response the client already has."""
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO analysis_log (ts, filename, tree_count, canopy_m2, cover_pct, co2e_tons, inference_time_ms, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    filename, tree_count, canopy_m2, cover_pct, co2e_tons, inference_time_ms, source
                )
            )
            limit = hardware.CONFIG["analytics_history_limit"]
            conn.execute(
                "DELETE FROM analysis_log WHERE id NOT IN "
                "(SELECT id FROM analysis_log ORDER BY id DESC LIMIT ?)",
                (limit,)
            )
    except Exception:
        # Analytics logging must never break the actual analysis response.
        logger.exception("Failed to log analysis to analytics DB")


def get_history(limit: int = 100) -> List[Dict[str, Any]]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM analysis_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_global_stats() -> Dict[str, Any]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS total_runs, "
            "COALESCE(SUM(tree_count), 0) AS total_trees, "
            "COALESCE(SUM(canopy_m2), 0) AS total_canopy_m2, "
            "COALESCE(SUM(co2e_tons), 0) AS total_co2e_tons, "
            "COALESCE(AVG(inference_time_ms), 0) AS avg_inference_time_ms "
            "FROM analysis_log"
        ).fetchone()
        return {
            "total_runs": row[0],
            "total_trees": row[1],
            "total_canopy_m2": round(row[2], 2),
            "total_canopy_ha": round(row[2] / 10000.0, 4),
            "total_co2e_tons": round(row[3], 2),
            "avg_inference_time_ms": round(row[4], 1)
        }


def get_leaderboard(limit: int = 20) -> List[Dict[str, Any]]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT filename, MAX(cover_pct) AS cover_pct, MAX(co2e_tons) AS co2e_tons, MAX(tree_count) AS tree_count "
            "FROM analysis_log GROUP BY filename ORDER BY co2e_tons DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def export_all() -> Dict[str, Any]:
    return {
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "global_stats": get_global_stats(),
        "history": get_history(limit=hardware.CONFIG["analytics_history_limit"])
    }
