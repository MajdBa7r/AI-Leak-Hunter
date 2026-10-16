"""Append-only investigation decision log (SQLite, Python standard library).

Every decision is a new row - nothing is updated or deleted - so the log is a
complete audit trail: who concluded what, when, why, and what was recovered.
The current status of a case is its latest row.

Statuses distinguish what the system flagged from what a human concluded:
    NEW                 not yet reviewed (implicit, never stored)
    IN_REVIEW           investigator working on it
    NEEDS_INFORMATION   waiting for records (supplier statement, GRN, contract ...)
    CONFIRMED_LEAKAGE   reviewer confirmed money was or would be lost
    NO_ISSUE            reviewer found a legitimate explanation (false positive)
    ESCALATED           handed to audit / legal / fraud team
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

STATUSES = ["NEW", "IN_REVIEW", "NEEDS_INFORMATION", "CONFIRMED_LEAKAGE", "NO_ISSUE", "ESCALATED"]
CLOSED = {"CONFIRMED_LEAKAGE", "NO_ISSUE"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    decision_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id      TEXT NOT NULL,
    status           TEXT NOT NULL,
    reviewer         TEXT NOT NULL,
    note             TEXT NOT NULL DEFAULT '',
    recovered_amount REAL NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_decisions_incident ON decisions(incident_id);
CREATE TRIGGER IF NOT EXISTS no_update BEFORE UPDATE ON decisions
BEGIN SELECT RAISE(ABORT, 'decision log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS no_delete BEFORE DELETE ON decisions
BEGIN SELECT RAISE(ABORT, 'decision log is append-only'); END;
"""


def default_db_path(out_dir: str | Path = "outputs") -> Path:
    return Path(os.environ.get("LEAK_HUNTER_CASE_DB", Path(out_dir) / "case_decisions.sqlite"))


class CaseStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self):
        return sqlite3.connect(self.path)

    def record(self, incident_id: str, status: str, reviewer: str, note: str = "",
               recovered_amount: float = 0.0) -> int:
        if status not in STATUSES or status == "NEW":
            raise ValueError(f"status must be one of {STATUSES[1:]}")
        if not reviewer or not reviewer.strip():
            raise ValueError("reviewer is required")
        if recovered_amount < 0:
            raise ValueError("recovered_amount must be >= 0")
        if recovered_amount > 0 and status != "CONFIRMED_LEAKAGE":
            raise ValueError("a recovered amount can only be recorded with CONFIRMED_LEAKAGE")
        with self._conn() as c:
            cur = c.execute("INSERT INTO decisions (incident_id, status, reviewer, note, recovered_amount, created_at) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            (incident_id, status, reviewer.strip(), note.strip(), round(float(recovered_amount), 2),
                             datetime.now(timezone.utc).isoformat(timespec="seconds")))
            return int(cur.lastrowid)

    def history(self, incident_id: str | None = None) -> pd.DataFrame:
        q, args = "SELECT * FROM decisions", ()
        if incident_id is not None:
            q, args = q + " WHERE incident_id = ?", (incident_id,)
        with self._conn() as c:
            return pd.read_sql_query(q + " ORDER BY decision_id", c, params=args)

    def latest(self) -> pd.DataFrame:
        h = self.history()
        if h.empty:
            return pd.DataFrame(columns=["incident_id", "status", "reviewer", "note", "recovered_amount", "created_at"])
        return h.sort_values("decision_id").groupby("incident_id").tail(1).set_index("incident_id")

    def status_of(self, incident_ids) -> pd.Series:
        lat = self.latest()
        return pd.Series([lat.status.get(i, "NEW") if len(lat) else "NEW" for i in incident_ids], index=list(incident_ids))

    def summary(self, incidents: pd.DataFrame) -> dict:
        """Operational metrics: how the queue is performing according to reviewers."""
        st = self.status_of(incidents.incident_id)
        lat = self.latest()
        closed = st[st.isin(CLOSED)]
        confirmed = st[st == "CONFIRMED_LEAKAGE"].index
        return {
            "cases": len(incidents), "reviewed": int((st != "NEW").sum()), "closed": len(closed),
            "confirmed_leakage": int((st == "CONFIRMED_LEAKAGE").sum()), "no_issue": int((st == "NO_ISSUE").sum()),
            "reviewer_confirmed_precision": round(len(confirmed) / len(closed), 4) if len(closed) else None,
            "recovered_amount": round(float(lat.loc[lat.index.isin(confirmed), "recovered_amount"].sum()), 2) if len(lat) else 0.0,
            "by_status": st.value_counts().to_dict(),
        }
