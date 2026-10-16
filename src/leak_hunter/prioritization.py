"""Case prioritization, linked cases and supplier profiles.

Risk score (0-100, points-based, every component shown to the investigator):

  exposure   up to 35   35 x log10(1 + de-duplicated potential exposure) / log10(1 + 50,000)
  evidence   up to 30   strongest rule finding with a contractual basis, by confidence
                        (HIGH 30 / MEDIUM 20 / LOW 8); statistical or historical anomaly
                        without a contractual basis ("unusual only") 8
  realization up to 15  confirmed duplicate payment 15; otherwise exposure already paid 7
  anomaly    up to 20   Isolation Forest: 20 x clip((p - 0.90) / 0.10, 0, 1), p = percentile of
                        the line's score among TRAINING-window scores (calibration uses no
                        labels and no test data)

Bands: P1 >= 60 "Investigate now", P2 >= 35 "Review this cycle", P3 "Monitor".

Expected value (money-first ordering): de-duplicated exposure x confidence factor
(HIGH 1.0 / MEDIUM 0.6 / LOW 0.25; no contractual basis 0), ties by risk score.
``priority_rank`` follows ``PrioritizationConfig.queue_order``; ``risk_rank`` and
``value_rank`` are always both provided.

The score ranks cases for attention. It is not a probability of fraud and it never
changes a finding's exposure. Weights are business judgement fixed before the queue
evaluation (reports_queue/) was run; they were not fitted to labels.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import DetectionConfig

EVIDENCE_TIERS = {"RULE_VIOLATION": "Rule violation with contractual basis",
                  "UNUSUAL_PATTERN": "Unusual pattern, no contractual basis"}


def training_percentiles(scores: pd.DataFrame) -> pd.Series:
    """Percentile of each line's anomaly score within the TRAINING-window scores."""
    ref = np.sort(scores.loc[scores.in_training_window, "anomaly_score"].to_numpy())
    p = np.searchsorted(ref, scores.anomaly_score.to_numpy(), "right") / max(1, len(ref))
    return pd.Series(p, index=scores.invoice_line_id.to_numpy())


def _confidence(details: str) -> str:
    try:
        return json.loads(details).get("confidence", "HIGH")
    except (TypeError, ValueError):
        return "HIGH"


def score_incidents(incidents: pd.DataFrame, findings: pd.DataFrame, scores: pd.DataFrame | None,
                    config: DetectionConfig | None = None) -> pd.DataFrame:
    """Add risk_score, its components, band and evidence tier; sort by risk."""
    cfg = (config or DetectionConfig()).prioritization
    if incidents.empty:
        return incidents
    inc = incidents.copy()
    pct = training_percentiles(scores) if scores is not None else pd.Series(dtype=float)
    rb = findings[findings.detection_method == "RULE_BASED"].copy()
    rb["conf"] = rb.details.map(_confidence)
    # a rule finding carries contractual evidence when its incident has exposure, except
    # zero-exposure historical spikes (unusual price, no contractual basis)
    rb["contractual"] = ~((rb.finding_type == "HISTORICAL_PRICE_SPIKE") & (rb.potential_exposure == 0))
    order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    best = (rb[rb.contractual].assign(o=lambda d: d.conf.map(order)).sort_values("o")
            .groupby("incident_id").conf.last())
    lines = inc.invoice_line_ids.str.split("|")
    inc["if_percentile"] = lines.map(lambda ls: float(max((pct.get(l, 0.0) for l in ls), default=0.0)))
    inc["rule_confidence"] = inc.incident_id.map(best)
    has_contract = inc.rule_confidence.notna() & (inc.potential_exposure > 0)
    inc["evidence_tier"] = np.where(has_contract, "RULE_VIOLATION", "UNUSUAL_PATTERN")

    e = inc.potential_exposure.clip(lower=0)
    inc["pts_exposure"] = cfg.exposure_points * np.minimum(1.0, np.log10(1 + e) / np.log10(1 + cfg.exposure_cap))
    inc["pts_evidence"] = np.where(has_contract, inc.rule_confidence.map(cfg.evidence_points).fillna(0.0),
                                   cfg.evidence_points["UNUSUAL_ONLY"])
    inc["pts_realization"] = np.where(inc.confirmed_loss > 0, cfg.confirmed_loss_points,
                                      np.where(inc.paid_exposure > 0, cfg.paid_exposure_points, 0.0))
    inc["pts_anomaly"] = cfg.anomaly_points * np.clip((inc.if_percentile - cfg.anomaly_floor_percentile)
                                                      / (1 - cfg.anomaly_floor_percentile), 0, 1)
    comp = ["pts_exposure", "pts_evidence", "pts_realization", "pts_anomaly"]
    inc[comp] = inc[comp].round(1)
    inc["risk_score"] = inc[comp].sum(axis=1).round(1)
    inc["band"] = inc.risk_score.map(lambda s: next(b for t, b in cfg.bands if s >= t))
    inc["band_label"] = inc.band.map(cfg.band_labels)
    inc["expected_value"] = np.where(has_contract, inc.potential_exposure
                                     * inc.rule_confidence.map(cfg.confidence_value_factor).fillna(0.0), 0.0).round(2)
    inc = inc.rename(columns={"priority_rank": "severity_rank"})
    inc["risk_rank"] = inc.risk_score.rank(ascending=False, method="first").astype(int)
    ev_order = inc.sort_values(["expected_value", "risk_score", "incident_id"], ascending=[False, False, True]).index
    inc.loc[ev_order, "value_rank"] = np.arange(1, len(inc) + 1)
    inc["value_rank"] = inc.value_rank.astype(int)
    key = ["risk_score", "potential_exposure"] if cfg.queue_order == "risk_score" else ["expected_value", "risk_score"]
    inc = inc.sort_values(key + ["incident_id"], ascending=[False, False, True])
    inc.insert(1, "priority_rank", np.arange(1, len(inc) + 1))
    return inc.reset_index(drop=True)


def link_incidents(incidents: pd.DataFrame, findings: pd.DataFrame) -> pd.DataFrame:
    """related_incident_ids: other incidents that share a PO line or an invoice document."""
    inc = incidents.copy()
    f = findings[["incident_id", "po_line_id", "invoice_id"]].drop_duplicates()
    keys = {}
    for r in f.itertuples():
        for k in (("PO", r.po_line_id), ("INV", r.invoice_id)):
            if isinstance(k[1], str):
                keys.setdefault(k, set()).add(r.incident_id)
    rel = {i: set() for i in inc.incident_id}
    for ids in keys.values():
        if len(ids) > 1:
            for i in ids:
                rel[i] |= ids - {i}
    inc["related_incident_ids"] = inc.incident_id.map(lambda i: "|".join(sorted(rel.get(i, ()))))
    return inc


def supplier_profiles(view: pd.DataFrame, incidents: pd.DataFrame, suppliers: pd.DataFrame) -> pd.DataFrame:
    """Per-supplier activity and risk summary (navigation aid, not a detector)."""
    act = view.groupby("supplier_id").agg(invoices=("invoice_id", "nunique"), invoice_lines=("invoice_line_id", "size"),
                                          spend=("line_amount", "sum"), first_invoice=("invoice_date", "min"),
                                          last_invoice=("invoice_date", "max"))
    if len(incidents):
        inc_dates = view.set_index("invoice_line_id").invoice_date
        x = incidents.assign(month=incidents.invoice_line_ids.str.split("|").map(
            lambda ls: max(inc_dates[l] for l in ls).strftime("%Y-%m")))
        g = x.groupby("supplier_id").agg(incidents=("incident_id", "size"),
                                         p1_incidents=("band", lambda s: int((s == "P1").sum())),
                                         potential_exposure=("potential_exposure", "sum"),
                                         paid_exposure=("paid_exposure", "sum"),
                                         confirmed_loss=("confirmed_loss", "sum"),
                                         max_risk_score=("risk_score", "max"),
                                         months_with_incidents=("month", "nunique"),
                                         incident_types=("incident_type", lambda s: "|".join(sorted(set(s)))))
        act = act.join(g, how="left")
    for c in ["incidents", "p1_incidents", "potential_exposure", "paid_exposure", "confirmed_loss", "max_risk_score",
              "months_with_incidents"]:
        if c not in act:
            act[c] = 0
        act[c] = act[c].fillna(0)
    act["incident_types"] = act.get("incident_types", pd.Series(dtype=str)).fillna("")
    act["exposure_share_of_spend"] = (act.potential_exposure / act.spend).round(4)
    act["recurring_issues"] = (act.incidents >= 3) & (act.months_with_incidents >= 2)
    out = suppliers.set_index("supplier_id").join(act, how="right").reset_index()
    for c in ["spend", "potential_exposure", "paid_exposure", "confirmed_loss"]:
        out[c] = out[c].round(2)
    return out.sort_values(["potential_exposure", "max_risk_score"], ascending=False).reset_index(drop=True)
