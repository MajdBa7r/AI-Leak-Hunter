"""Common findings schema shared by the rules and the Isolation Forest.

One row = one finding = one (detector, invoice line) pair. Identifiers are
stable across runs on the same data:

    finding_id   F-<CODE>-<invoice_line_id>   CODE in DUP, QTY, PRC, HPS, IFR
    incident_id  INC-DUP-<supplier_id>-<normalised number> for duplicate groups,
                 otherwise INC-<invoice_line_id>

All findings on the same invoice line (and both entries of a duplicate pair)
share one incident_id, so reports count incidents and de-duplicated exposure
from ``exposure.incident_table`` instead of summing findings.

``evidence_record_ids`` and ``details`` are JSON objects (strings in CSV):
evidence is ``{"invoice_lines": [...], "invoices": [...], "po_lines": [...],
"receipts": [...], "amendments": [...], "suppliers": [...]}``; details hold the
observed and reference values the finding is based on.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np
import pandas as pd

from .config import DetectionConfig
from .exposure import DUP, HIST, PRICE, QTY, UNUSUAL

FINDING_COLUMNS = [
    "finding_id", "incident_id", "finding_type", "detection_method", "rule_id",
    "invoice_id", "invoice_line_id", "invoice_number", "supplier_id", "po_id", "po_line_id", "item_id",
    "invoice_date", "severity", "potential_exposure", "paid_exposure", "confirmed_loss", "amount_paid",
    "line_amount", "currency", "exposure_basis", "evidence_record_ids", "details", "anomaly_score", "risk_rank",
    "related_finding_ids", "description", "status",
]
FINDING_TYPES = {DUP, QTY, PRICE, HIST, UNUSUAL}
TYPE_CODES = {DUP: "DUP", QTY: "QTY", PRICE: "PRC", HIST: "HPS", UNUSUAL: "IFR"}
METHODS = {"RULE_BASED", "ISOLATION_FOREST"}
SEVERITIES = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
STATUSES = {"OPEN", "UNDER_REVIEW", "CONFIRMED", "DISMISSED"}
EVIDENCE_KEYS = {"invoice_lines", "invoices", "po_lines", "receipts", "amendments", "suppliers"}


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.strftime("%Y-%m-%d")
    if isinstance(o, (np.ndarray, tuple, set)):
        return list(o)
    if isinstance(o, float) and math.isnan(o):
        return None
    return str(o)


def to_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=_jsonable)


def _incident_id(line_id: str, dup_key) -> str:
    if isinstance(dup_key, str):
        return "INC-DUP-" + re.sub(r"[^0-9A-Za-z]+", "-", dup_key).strip("-")
    return f"INC-{line_id}"


def assign_severity(h: pd.DataFrame, config: DetectionConfig) -> pd.Series:
    s = config.severity
    exp = h.potential_exposure.astype(float)
    sev = np.where(exp >= s.high_exposure, "HIGH", np.where(exp >= s.medium_exposure, "MEDIUM", "LOW"))
    sev = pd.Series(sev, index=h.index, dtype="object")
    role = h.details.map(lambda d: d.get("role") if isinstance(d, dict) else None)
    dup = h.finding_type == DUP
    sev[dup & (role == "REPEAT")] = "HIGH"
    sev[dup & (role == "REPEAT") & (h.confirmed_loss > 0)] = "CRITICAL"
    sev[dup & (role == "ORIGINAL")] = "MEDIUM"
    ifr = h.finding_type == UNUSUAL
    pct = h.details.map(lambda d: d.get("score_percentile", 0) if isinstance(d, dict) else 0)
    sev[ifr] = np.where(pct[ifr] >= s.if_high_percentile, "MEDIUM", "LOW")
    return sev


def build_findings(hits: pd.DataFrame, view: pd.DataFrame, config: DetectionConfig | None = None,
                   method: str = "RULE_BASED") -> pd.DataFrame:
    """Convert rule or Isolation Forest hits into the common findings schema."""
    cfg = config or DetectionConfig()
    if hits.empty:
        return pd.DataFrame(columns=FINDING_COLUMNS)
    ctx = view.set_index("invoice_line_id")[["invoice_id", "invoice_number", "supplier_id", "po_id", "po_line_id",
                                             "item_id", "invoice_date", "amount_paid", "line_amount",
                                             "dup_group_key"]]
    missing = set(hits.invoice_line_id) - set(ctx.index)
    if missing:
        raise ValueError(f"Hits reference unknown invoice lines: {sorted(missing)[:5]}")
    f = hits.join(ctx, on="invoice_line_id")
    f["finding_id"] = "F-" + f.finding_type.map(TYPE_CODES) + "-" + f.invoice_line_id
    f["incident_id"] = [_incident_id(l, k) for l, k in zip(f.invoice_line_id, f.dup_group_key)]
    f["detection_method"] = method
    f["severity"] = assign_severity(f, cfg)
    f["currency"] = cfg.currency
    f["status"] = "OPEN"
    f["invoice_date"] = f.invoice_date.dt.strftime("%Y-%m-%d")
    f["anomaly_score"] = f.details.map(lambda d: d.get("anomaly_score", np.nan))
    f["risk_rank"] = f.details.map(lambda d: d.get("risk_rank", np.nan)).astype("Int64")
    for c in ["amount_paid", "line_amount", "potential_exposure", "paid_exposure", "confirmed_loss"]:
        f[c] = f[c].astype(float).round(2)
    f["evidence_record_ids"] = f.evidence.map(lambda e: to_json({k: list(v) for k, v in e.items() if v}))
    f["details"] = f.details.map(to_json)
    f["related_finding_ids"] = "[]"
    return f[FINDING_COLUMNS].reset_index(drop=True)


def link_related(findings: pd.DataFrame) -> pd.DataFrame:
    """Fill related_finding_ids with the other findings of the same incident."""
    f = findings.copy()
    by_inc = f.groupby("incident_id").finding_id.apply(list)
    f["related_finding_ids"] = [to_json(sorted(x for x in by_inc[i] if x != fid))
                                for i, fid in zip(f.incident_id, f.finding_id)]
    return f


def validate_findings(f: pd.DataFrame) -> None:
    """Raise ValueError if ``f`` does not conform to the common schema."""
    errs = []
    if list(f.columns) != FINDING_COLUMNS:
        errs.append(f"columns differ from schema: {set(FINDING_COLUMNS) ^ set(f.columns)}")
    else:
        if f.finding_id.isna().any() or not f.finding_id.is_unique:
            errs.append("finding_id must be unique and non-null")
        for col, allowed in [("finding_type", FINDING_TYPES), ("detection_method", METHODS),
                             ("severity", SEVERITIES), ("status", STATUSES)]:
            bad = set(f[col]) - allowed
            if bad:
                errs.append(f"{col} has invalid values {bad}")
        for c in ["potential_exposure", "paid_exposure", "confirmed_loss"]:
            x = f[c].astype(float)
            if not np.isfinite(x).all() or (x < 0).any():
                errs.append(f"{c} must be finite and >= 0")
            if not np.allclose(x * 100, np.round(x * 100), atol=1e-6):
                errs.append(f"{c} must have at most two decimals")
        if (f.paid_exposure > f.potential_exposure + 0.005).any():
            errs.append("paid_exposure exceeds potential_exposure")
        if (f.confirmed_loss > f.paid_exposure + 0.005).any():
            errs.append("confirmed_loss exceeds paid_exposure")
        if (f.potential_exposure[f.detection_method == "ISOLATION_FOREST"] != 0).any():
            errs.append("Isolation Forest findings must not carry monetary exposure")
        for i, (ev, line) in enumerate(zip(f.evidence_record_ids, f.invoice_line_id)):
            try:
                e = json.loads(ev)
                json.loads(f.details.iloc[i])
            except (TypeError, ValueError):
                errs.append(f"row {i}: evidence/details are not valid JSON")
                break
            if not isinstance(e, dict) or set(e) - EVIDENCE_KEYS or line not in e.get("invoice_lines", []):
                errs.append(f"row {i}: evidence must be a dict of ID lists including the invoice line")
                break
    if errs:
        raise ValueError("Findings schema violations: " + "; ".join(errs))
