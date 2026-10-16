"""Point-in-time historical baselines (no future information).

Every statistic for an invoice line dated D uses only observations with an
invoice date strictly before D (same-day lines are excluded, because their
order within a day is not reliable). These functions back both Rule D and the
Isolation Forest features.

Valid price observations: positive unit price, a known PO line, and (by
default) not a duplicate re-entry, i.e. an invoice whose supplier+number was
already seen earlier is not counted as a new purchase.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DetectionConfig

MAD_TO_SIGMA = 1.4826


def _valid_price_mask(view: pd.DataFrame, exclude_reentries: bool) -> np.ndarray:
    ok = (view.unit_price > 0) & view.po_line_found
    if exclude_reentries:
        ok &= view.dup_rank == 0
    return ok.to_numpy()


def price_history(view: pd.DataFrame, config: DetectionConfig | None = None) -> pd.DataFrame:
    """Prior-price baseline per invoice line for the same supplier and item.

    Returns columns (indexed like ``view``): hist_n_prior, hist_baseline,
    hist_mad, hist_robust_cv, hist_ratio, hist_prior_line_ids, hist_prior_prices,
    hist_prior_min, hist_prior_max, hist_last_price.
    """
    cfg = (config or DetectionConfig()).rules
    valid = pd.Series(_valid_price_mask(view, cfg.hist_exclude_duplicate_reentries), index=view.index)
    out = {k: {} for k in ["hist_n_prior", "hist_baseline", "hist_mad", "hist_prior_line_ids", "hist_prior_prices",
                           "hist_prior_min", "hist_prior_max", "hist_last_price"]}
    for _, g in view.groupby(["supplier_id", "item_id"], sort=False):
        g = g.sort_values(["invoice_date", "invoice_id", "invoice_line_id"])
        d = g.invoice_date.to_numpy()
        p = g.unit_price.to_numpy(dtype=float)
        ids = g.invoice_line_id.to_numpy()
        ok = valid.loc[g.index].to_numpy()
        for j, ix in enumerate(g.index):
            m = (d < d[j]) & ok
            n = int(m.sum())
            out["hist_n_prior"][ix] = n
            out["hist_prior_line_ids"][ix] = list(ids[m])
            out["hist_prior_prices"][ix] = [round(float(x), 2) for x in p[m]]
            if n:
                pp = p[m]
                base = float(np.median(pp)) if cfg.hist_baseline == "median" else float(np.mean(pp))
                out["hist_baseline"][ix] = base
                out["hist_mad"][ix] = float(np.median(np.abs(pp - np.median(pp))))
                out["hist_prior_min"][ix], out["hist_prior_max"][ix] = float(pp.min()), float(pp.max())
                out["hist_last_price"][ix] = float(pp[-1])
    h = pd.DataFrame({k: pd.Series(v, dtype="object" if k in ("hist_prior_line_ids", "hist_prior_prices")
                                   else float) for k, v in out.items()}).reindex(view.index)
    h["hist_n_prior"] = h.hist_n_prior.fillna(0).astype(int)
    for k in ("hist_prior_line_ids", "hist_prior_prices"):
        h[k] = h[k].apply(lambda x: x if isinstance(x, list) else [])
    with np.errstate(divide="ignore", invalid="ignore"):
        h["hist_robust_cv"] = MAD_TO_SIGMA * h.hist_mad / h.hist_baseline
        h["hist_ratio"] = view.unit_price / h.hist_baseline
    if cfg.hist_adaptive:
        h["hist_effective_threshold"] = np.maximum(cfg.hist_ratio_threshold,
                                                   1 + cfg.hist_dispersion_k * h.hist_robust_cv.fillna(0))
    else:
        h["hist_effective_threshold"] = cfg.hist_ratio_threshold
    h["hist_sufficient"] = h.hist_n_prior >= cfg.hist_min_prior
    return h


def supplier_amount_history(view: pd.DataFrame, min_prior: int = 5) -> pd.DataFrame:
    """Median line amount of the supplier's strictly earlier invoice lines (any item)."""
    med, n = {}, {}
    for _, g in view.groupby("supplier_id", sort=False):
        d = g.invoice_date.to_numpy()
        a = g.line_amount.to_numpy(dtype=float)
        for j, ix in enumerate(g.index):
            m = d < d[j]
            n[ix] = int(m.sum())
            med[ix] = float(np.median(a[m])) if m.sum() >= min_prior else np.nan
    return pd.DataFrame({"supplier_prior_lines": pd.Series(n), "supplier_prior_median_amount": pd.Series(med)}
                        ).reindex(view.index)


def supplier_invoice_frequency(view: pd.DataFrame, window_days: int = 30) -> pd.Series:
    """Number of distinct invoice documents from the same supplier dated in
    [D - window, D) - i.e. strictly before the line's invoice date."""
    docs = view.drop_duplicates("invoice_id")[["invoice_id", "supplier_id", "invoice_date"]]
    res = {}
    for sup, g in docs.groupby("supplier_id"):
        d = np.sort(g.invoice_date.to_numpy())
        res[sup] = d
    w = np.timedelta64(window_days, "D")
    out = []
    for sup, dt in zip(view.supplier_id, view.invoice_date.to_numpy()):
        d = res[sup]
        out.append(int(np.searchsorted(d, dt, "left") - np.searchsorted(d, dt - w, "left")))
    return pd.Series(out, index=view.index, name="supplier_docs_prior_window")
