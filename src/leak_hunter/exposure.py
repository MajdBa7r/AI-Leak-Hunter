"""Money helpers and exposure aggregation.

All money is computed in integer cents and converted back with two decimals,
so sums never drift. Three amounts are kept separate on every finding:

potential_exposure   amount that may be recoverable if the finding is confirmed
                     (formula per rule, see detection_rules.py)
paid_exposure        part of potential_exposure that has already been paid out
                     (prorated by amount_paid / line_amount; for duplicates, the
                     payment beyond the original amount)
confirmed_loss       only where payment records alone prove a loss: a duplicate
                     payment (both entries paid beyond the original amount).
                     Price and quantity overpayments stay "paid exposure" until
                     an investigator confirms them (credit notes, late receipts).

Overlap handling (one invoice line can trigger several rules):
    price component    = max(PO_PRICE_MISMATCH, HISTORICAL_PRICE_SPIKE exposure)
                         - both measure the same contractual excess
    quantity component = QUANTITY_DISCREPANCY exposure (excess units at the
                         approved price; additive to the price component)
    duplicate component= DUPLICATE_INVOICE exposure (whole repeated line amount;
                         it already contains any quantity/price excess on it)
    line exposure      = max(duplicate, quantity + price)
Incident exposure is the sum of its lines' exposures. Isolation Forest findings
carry no monetary exposure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DUP, QTY, PRICE, HIST, UNUSUAL = ("DUPLICATE_INVOICE", "QUANTITY_DISCREPANCY", "PO_PRICE_MISMATCH",
                                  "HISTORICAL_PRICE_SPIKE", "UNUSUAL_TRANSACTION")


def to_cents(x) -> np.ndarray | int:
    """Round USD amounts to integer cents (half away from zero)."""
    a = np.asarray(x, dtype=float)
    c = np.sign(a) * np.floor(np.abs(a) * 100 + 0.5 + 1e-9)
    return c.astype("int64") if c.ndim else int(c)


def from_cents(c) -> np.ndarray | float:
    a = np.asarray(c, dtype="int64") / 100.0
    return np.round(a, 2) if a.ndim else round(float(a), 2)


def prorate_paid(exposure_c, amount_paid_c, line_amount_c):
    """Portion of an exposure already paid, assuming payments apply pro rata to the line."""
    exposure_c, paid, amt = (np.asarray(v, dtype="int64") for v in (exposure_c, amount_paid_c, line_amount_c))
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(amt > 0, np.clip(paid / np.where(amt > 0, amt, 1), 0, 1), 0.0)
    out = np.floor(exposure_c * frac + 0.5 + 1e-9).astype("int64")
    return out if out.ndim else int(out)


def realized_duplicate_payments(amount_original_c: int, paid_sequence_c: list[int],
                                repeat_amounts_c: list[int]) -> list[int]:
    """Realized duplicate payment per repeat entry.

    ``paid_sequence_c`` is [paid on original, paid on repeat 1, paid on repeat 2, ...].
    Money paid across the group beyond the original amount is a duplicate payment;
    it is allocated to repeats in order and capped at each repeat's amount.
    """
    out, cum, realized_total = [], paid_sequence_c[0], 0
    for paid, cap in zip(paid_sequence_c[1:], repeat_amounts_c):
        cum += paid
        r = max(0, min(cap, cum - amount_original_c - realized_total))
        out.append(int(r))
        realized_total += r
    return out


def line_exposure(findings: pd.DataFrame) -> pd.DataFrame:
    """De-duplicated exposure per invoice line from rule-based findings."""
    rb = findings[findings.detection_method == "RULE_BASED"]
    cols = ["potential_exposure", "paid_exposure", "confirmed_loss"]
    if rb.empty:
        return pd.DataFrame(columns=["invoice_line_id", "line_exposure", "line_paid_exposure",
                                     "line_confirmed_loss", "line_types"])
    c = rb[["invoice_line_id", "finding_type"]].copy()
    for k in cols:
        c[k] = to_cents(rb[k].to_numpy())
    p = c.pivot_table(index="invoice_line_id", columns="finding_type", values=cols, aggfunc="max", fill_value=0)

    def get(k, t):
        return p[(k, t)] if (k, t) in p.columns else pd.Series(0, index=p.index)

    out = pd.DataFrame(index=p.index)
    for k, name in [("potential_exposure", "line_exposure"), ("paid_exposure", "line_paid_exposure")]:
        dup = get(k, DUP)
        price = np.maximum(get(k, PRICE), get(k, HIST))
        other = get(k, QTY) + price
        out[name] = np.where(get("potential_exposure", DUP) >= get("potential_exposure", QTY)
                             + np.maximum(get("potential_exposure", PRICE), get("potential_exposure", HIST)),
                             dup, other)
    out["line_confirmed_loss"] = get("confirmed_loss", DUP)
    for k in ["line_exposure", "line_paid_exposure", "line_confirmed_loss"]:
        out[k] = from_cents(out[k].to_numpy().astype("int64"))
    out["line_types"] = rb.groupby("invoice_line_id").finding_type.apply(lambda s: "|".join(sorted(set(s))))
    return out.reset_index()


SEVERITY_ORDER = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}


def incident_table(findings: pd.DataFrame) -> pd.DataFrame:
    """One row per incident with de-duplicated exposure (safe to sum)."""
    if findings.empty:
        return pd.DataFrame()
    le = line_exposure(findings).set_index("invoice_line_id")
    rows = []
    for inc, g in findings.groupby("incident_id", sort=False):
        lines = sorted(set(g.invoice_line_id))
        rb = g[g.detection_method == "RULE_BASED"]
        exp = le.reindex([x for x in lines if x in le.index])
        rows.append({
            "incident_id": inc,
            "incident_type": _incident_type(rb, g),
            "finding_types": "|".join(sorted(set(g.finding_type))),
            "detection_methods": "|".join(sorted(set(g.detection_method))),
            "supplier_id": g.supplier_id.iloc[0],
            "invoice_ids": "|".join(sorted(set(g.invoice_id))),
            "invoice_line_ids": "|".join(lines),
            "n_findings": len(g),
            "finding_ids": "|".join(g.finding_id),
            "max_severity": max(g.severity, key=lambda s: SEVERITY_ORDER.get(s, 0)),
            "potential_exposure": round(float(exp.line_exposure.sum()), 2) if len(exp) else 0.0,
            "paid_exposure": round(float(exp.line_paid_exposure.sum()), 2) if len(exp) else 0.0,
            "confirmed_loss": round(float(exp.line_confirmed_loss.sum()), 2) if len(exp) else 0.0,
            "max_anomaly_score": g.anomaly_score.max() if g.anomaly_score.notna().any() else np.nan,
            "currency": g.currency.iloc[0],
            "status": "OPEN",
        })
    inc = pd.DataFrame(rows)
    inc["_sev"] = inc.max_severity.map(SEVERITY_ORDER)
    inc = inc.sort_values(["_sev", "potential_exposure", "max_anomaly_score"], ascending=[False, False, False],
                          na_position="last").drop(columns="_sev").reset_index(drop=True)
    inc.insert(1, "priority_rank", np.arange(1, len(inc) + 1))
    return inc


def _incident_type(rb: pd.DataFrame, g: pd.DataFrame) -> str:
    for t in [DUP, QTY, PRICE, HIST]:
        if (rb.finding_type == t).any():
            return t
    return UNUSUAL
