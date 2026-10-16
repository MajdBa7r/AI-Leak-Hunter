"""Unsupervised anomaly detection with scikit-learn's Isolation Forest.

Purpose: surface invoice lines that look statistically unusual even when no
deterministic rule fires. It is a triage aid, not a fraud classifier.

Features (all numeric, all derivable from procurement data at audit time):

log_line_amount                  log(1 + quantity x unit price)
price_vs_approved_rel            (unit price - approved price) / approved price
share_billed_ahead_of_receipt    max(0, cumulative invoiced - received AS OF the
                                 invoice date) / cumulative invoiced, in [0, 1].
                                 One-directional (over-billing only; under-billing
                                 scores 0) and point-in-time (receipts dated after
                                 the invoice are not visible).
hist_price_log_ratio             log(unit price / median of EARLIER prices, same
                                 supplier + item); NaN with < hist_min_prior
hist_price_robust_z              (price - median) / max(1.4826 x MAD, 1% of median)
has_price_history                1 when the historical baseline exists, else 0
log_amount_vs_supplier_median    log(line amount / median EARLIER line amount of
                                 the supplier); NaN with too little history
log_supplier_docs_prior_window   log(1 + supplier invoices in the prior N days)
days_po_to_invoice               invoice date - PO date

Excluded on purpose: identifiers (invoice/PO/supplier/item IDs, invoice
numbers), names, raw categorical IDs, and every ground-truth column
(``data_loader.LABEL_COLUMNS``). ``assert_no_label_features`` enforces this.

Leakage control:
* historical features use strictly earlier invoice dates only (history.py);
* the quantity feature uses receipts dated on/before the invoice date only, and
  the approved price is the one in force on the invoice date;
* in "temporal" mode the imputer and the forest are fitted on the earliest
  ``train_fraction`` of invoice lines only; later lines are scored out-of-time;
* the review threshold is the (1 - review_budget) quantile of TRAINING scores;
* no labels are read anywhere in this module.

Known limitation: split points are drawn within the training range, so a value
beyond it is isolated no faster than the most extreme training value, and a
feature that is constant in the training window cannot contribute.

Score direction: ``anomaly_score = -IsolationForest.score_samples(x)``, roughly
in (0, 1); HIGHER = more isolated / more unusual. It is a relative ranking
score, not a probability of fraud or of financial loss.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .config import DetectionConfig
from .data_loader import LABEL_COLUMNS
from .exposure import UNUSUAL
from .history import MAD_TO_SIGMA, supplier_amount_history, supplier_invoice_frequency

FEATURE_COLUMNS = ["log_line_amount", "price_vs_approved_rel", "share_billed_ahead_of_receipt",
                   "hist_price_log_ratio", "hist_price_robust_z", "has_price_history",
                   "log_amount_vs_supplier_median", "log_supplier_docs_prior_window", "days_po_to_invoice"]
FORBIDDEN_FEATURE_TOKENS = ("_id", "invoice_number", "supplier_name", "item_description", "anomaly", "label",
                            "ground", "truth", "expected_", "scenario", "duplicate_group")


class FeatureLeakageError(ValueError):
    pass


def assert_no_label_features(features: pd.DataFrame) -> None:
    bad = [c for c in features.columns if c in LABEL_COLUMNS or any(t in c for t in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise FeatureLeakageError(f"Forbidden columns in model features: {bad}")
    non_num = [c for c in features.columns if not pd.api.types.is_numeric_dtype(features[c])]
    if non_num:
        raise FeatureLeakageError(f"Non-numeric model features: {non_num}")


def build_features(view: pd.DataFrame, history: pd.DataFrame, config: DetectionConfig | None = None) -> pd.DataFrame:
    """Numeric feature matrix indexed by invoice_line_id (point-in-time history only)."""
    cfg = config or DetectionConfig()
    ic = cfg.isolation_forest
    h = history.reindex(view.index)
    sup = supplier_amount_history(view, ic.supplier_history_min_prior)
    freq = supplier_invoice_frequency(view, ic.frequency_window_days)
    with np.errstate(divide="ignore", invalid="ignore"):
        X = pd.DataFrame(index=view.index)
        X["log_line_amount"] = np.log1p(view.line_amount.clip(lower=0))
        X["price_vs_approved_rel"] = (view.unit_price - view.approved_unit_price) / view.approved_unit_price
        ahead = (view.cum_invoiced - view.received_as_of_invoice).clip(lower=0)
        X["share_billed_ahead_of_receipt"] = (ahead / view.cum_invoiced.where(view.cum_invoiced > 0)).clip(0, 1)
        base = h.hist_baseline.where(h.hist_sufficient)
        X["hist_price_log_ratio"] = np.log(view.unit_price / base)
        scale = np.maximum(MAD_TO_SIGMA * h.hist_mad, 0.01 * h.hist_baseline).where(h.hist_sufficient)
        X["hist_price_robust_z"] = (view.unit_price - base) / scale
        X["has_price_history"] = h.hist_sufficient.astype(float)
        X["log_amount_vs_supplier_median"] = np.log(view.line_amount / sup.supplier_prior_median_amount)
        X["log_supplier_docs_prior_window"] = np.log1p(freq)
        X["days_po_to_invoice"] = (view.invoice_date - view.po_date).dt.days.astype(float)
    X.index = view.invoice_line_id.to_numpy()
    X.index.name = "invoice_line_id"
    X = X[FEATURE_COLUMNS].astype(float)
    assert_no_label_features(X)
    return X


def _replace_inf(a):
    a = np.asarray(a, dtype=float).copy()
    a[~np.isfinite(a)] = np.nan
    return a


def make_model(config: DetectionConfig | None = None) -> Pipeline:
    ic = (config or DetectionConfig()).isolation_forest
    return Pipeline([
        ("replace_inf", FunctionTransformer(_replace_inf, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("iforest", IsolationForest(n_estimators=ic.n_estimators, max_samples=ic.max_samples,
                                    max_features=ic.max_features, contamination=ic.contamination,
                                    random_state=ic.random_state, n_jobs=1)),
    ])


def temporal_train_mask(invoice_dates: pd.Series, train_fraction: float) -> np.ndarray:
    """True for lines dated before the cutoff date at ``train_fraction`` of the
    date-ordered lines (all lines of the cutoff date go to the scoring side)."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")
    d = np.sort(invoice_dates.to_numpy())
    cutoff = d[int(np.floor(len(d) * train_fraction))]
    return (invoice_dates < cutoff).to_numpy()


def run_isolation_forest(view: pd.DataFrame, history: pd.DataFrame, config: DetectionConfig | None = None,
                         features: pd.DataFrame | None = None) -> tuple[pd.DataFrame, Pipeline, pd.DataFrame]:
    """Fit (on the training window) and score every invoice line.

    Returns (scores, fitted_pipeline, features). ``scores`` columns:
    invoice_line_id, anomaly_score, risk_rank, score_percentile,
    in_training_window, review_threshold, above_review_threshold,
    model_default_outlier, top_features, plus the raw feature values.
    """
    cfg = config or DetectionConfig()
    ic = cfg.isolation_forest
    X = features if features is not None else build_features(view, history, cfg)
    assert_no_label_features(X)
    dates = pd.Series(view.invoice_date.to_numpy(), index=X.index)
    if ic.fit_mode == "temporal":
        train = temporal_train_mask(dates, ic.train_fraction)
    elif ic.fit_mode == "all":
        train = np.ones(len(X), dtype=bool)
    else:
        raise ValueError(f"Unknown fit_mode {ic.fit_mode!r}")
    if train.sum() < 50:
        raise ValueError(f"Training window has only {int(train.sum())} lines; increase train_fraction.")
    model = make_model(cfg).fit(X[train])
    score = -model.score_samples(X)
    train_scores = score[train]
    threshold = float(np.quantile(train_scores, 1 - ic.review_budget))

    # explanation: robust z of each (cleaned, imputed) feature vs the TRAINING distribution
    Xi = pd.DataFrame(model[:-1].transform(X), index=X.index, columns=FEATURE_COLUMNS)
    Xt = Xi[train]
    med, iqr = Xt.median(), (Xt.quantile(0.75) - Xt.quantile(0.25))
    sd = Xt.std(ddof=0)
    scale = iqr.where(iqr > 0, sd).where(lambda s: s > 0, 1.0)
    z = (Xi - med) / scale
    top = []
    for idx in X.index:
        zz = z.loc[idx].abs().sort_values(ascending=False).head(ic.explain_top_n)
        top.append([{"feature": f, "value": None if pd.isna(X.at[idx, f]) else round(float(X.at[idx, f]), 4),
                     "robust_z": round(float(z.at[idx, f]), 2)} for f in zz.index])

    s = pd.DataFrame({"invoice_line_id": X.index, "anomaly_score": np.round(score, 6)})
    s["risk_rank"] = s.anomaly_score.rank(ascending=False, method="first").astype(int)
    s["score_percentile"] = s.anomaly_score.rank(pct=True, method="max").round(6)
    s["in_training_window"] = train
    s["review_threshold"] = round(threshold, 6)
    s["above_review_threshold"] = s.anomaly_score >= threshold
    s["model_default_outlier"] = model.predict(X) == -1
    s["top_features"] = top
    s = pd.concat([s.reset_index(drop=True), X.reset_index(drop=True).add_prefix("feat_")], axis=1)
    return s, model, X


def isolation_forest_hits(scores: pd.DataFrame, view: pd.DataFrame,
                          rule_findings: pd.DataFrame | None = None) -> pd.DataFrame:
    """Hits for lines above the review threshold (common hit-table format).

    A line that already has rule findings is still reported (the statistical
    view is useful evidence) but it joins the rule's incident, so it is not
    counted as a new incident; ``details.overlaps_rule_finding`` says so.
    """
    from .detection_rules import HIT_COLUMNS
    sel = scores[scores.above_review_threshold]
    if sel.empty:
        return pd.DataFrame(columns=HIT_COLUMNS)
    vx = view.set_index("invoice_line_id")
    rule_by_line = (rule_findings.groupby("invoice_line_id").finding_id.apply(list)
                    if rule_findings is not None and len(rule_findings) else pd.Series(dtype=object))
    rows = []
    for r in sel.itertuples():
        v = vx.loc[r.invoice_line_id]
        overl = rule_by_line.get(r.invoice_line_id, [])
        feats = {c[5:]: (None if pd.isna(getattr(r, c)) else round(float(getattr(r, c)), 4))
                 for c in sel.columns if c.startswith("feat_")}
        details = {"anomaly_score": float(r.anomaly_score), "risk_rank": int(r.risk_rank),
                   "score_percentile": float(r.score_percentile), "review_threshold": float(r.review_threshold),
                   "in_training_window": bool(r.in_training_window), "top_features": r.top_features,
                   "feature_values": feats, "overlaps_rule_finding": bool(overl), "rule_finding_ids": overl,
                   "score_meaning": "higher = more unusual relative to the training window; not a probability"}
        drivers = ", ".join(f"{t['feature']} (z={t['robust_z']})" for t in r.top_features)
        desc = (f"Statistically unusual invoice line (Isolation Forest score {r.anomaly_score:.3f}, rank "
                f"{r.risk_rank}, above review threshold {r.review_threshold:.3f}). Main drivers: {drivers}. "
                + ("Also flagged by rule(s) " + ", ".join(overl) + "." if overl else
                   "No deterministic rule fired; review for context.") + " Not a fraud probability.")
        rows.append({"invoice_line_id": r.invoice_line_id, "finding_type": UNUSUAL, "rule_id": "ISOLATION_FOREST",
                     "potential_exposure": 0.0, "paid_exposure": 0.0, "confirmed_loss": 0.0,
                     "exposure_basis": "none: statistical outlier, no monetary formula",
                     "evidence": {"invoice_lines": [r.invoice_line_id], "invoices": [v.invoice_id],
                                  "po_lines": [v.po_line_id] if isinstance(v.po_line_id, str) else []},
                     "details": details, "description": desc, "incident_key": None})
    return pd.DataFrame(rows, columns=HIT_COLUMNS)
