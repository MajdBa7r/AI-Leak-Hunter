"""AI Leak Hunter - unsupervised anomaly-detection benchmark (5 experiments).

    PYTHONPATH=src python scripts/unsupervised_benchmark.py [--out reports_ml]

Protocol (leak-free by construction)
* Features: point-in-time only. Every historical statistic uses invoice lines
  dated STRICTLY before the line being scored; receipts count only if dated on
  or before the invoice. No IDs, names, labels or future timestamps.
* Temporal split: models, imputer, scaler and score calibration are fitted on
  the earliest 60% of invoice lines (by date). Metrics are computed on the
  later 40% (out-of-time). Labels (ground_truth.csv) are read only AFTER every
  model has scored the test window.
* No label-based tuning: hyper-parameters are fixed a priori, ensemble weights
  are equal, and the review threshold is the 98th percentile of TRAINING scores
  (a 2% review budget). The synthetic 10% anomaly rate is never used.

Experiments
  1  Isolation Forest (500 trees, 256 samples/tree)
  2a Local Outlier Factor (k = 35, novelty mode)
  2b ECOD (empirical-CDF tail probabilities, implemented here, no extra dependency)
  3  Robust Mahalanobis distance (Minimum Covariance Determinant = Elliptic Envelope)
  4  Autoencoder (scikit-learn MLP 32-8-32, L2 reconstruction error)
  5  Consensus ensemble: equal-weight average of the training-calibrated
     percentile ranks of the tree (1), distance (2a) and neural (4) scores

Outputs: reports_ml/benchmark.md, benchmark.json, roc_pr_curves.png,
         top1_attribution_waterfall.png, top_anomalies.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler

from leak_hunter.anomaly_detection import assert_no_label_features, build_features, temporal_train_mask
from leak_hunter.config import DetectionConfig
from leak_hunter.data_loader import load_procurement_data
from leak_hunter.history import price_history
from leak_hunter.reconciliation import build_line_view

SEED = 42
TRAIN_FRACTION = 0.60
REVIEW_BUDGET = 0.02
CLIP = 10.0                     # robust-scaled values clipped to +/-10 (fixed transform, no fitting)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------- features
def extra_history_features(view: pd.DataFrame) -> pd.DataFrame:
    """Additional point-in-time features (strictly earlier invoice dates only)."""
    out = pd.DataFrame(index=view.index)
    qty_ratio, amt_z, gap = {}, {}, {}
    for _, g in view.groupby(["supplier_id", "item_id"], sort=False):
        d, q = g.invoice_date.to_numpy(), g.quantity_invoiced.to_numpy(float)
        for j, ix in enumerate(g.index):
            m = d < d[j]
            qty_ratio[ix] = np.log(q[j] / np.median(q[m])) if m.sum() >= 3 else np.nan
    w = np.timedelta64(90, "D")
    for _, g in view.groupby("supplier_id", sort=False):
        d, a = g.invoice_date.to_numpy(), np.log1p(g.line_amount.to_numpy(float))
        docs = np.sort(g.drop_duplicates("invoice_id").invoice_date.to_numpy())
        for j, ix in enumerate(g.index):
            m = (d < d[j]) & (d >= d[j] - w)
            if m.sum() >= 5:
                med = np.median(a[m]); mad = np.median(np.abs(a[m] - med))
                amt_z[ix] = (a[j] - med) / max(1.4826 * mad, 0.05)
            else:
                amt_z[ix] = np.nan
            k = np.searchsorted(docs, d[j], "left")
            gap[ix] = np.log1p((d[j] - docs[k - 1]) / np.timedelta64(1, "D")) if k > 0 else np.nan
    out["log_qty_vs_prior_median_same_item"] = pd.Series(qty_ratio)
    out["supplier_90d_amount_robust_z"] = pd.Series(amt_z)
    out["log_days_since_prev_supplier_invoice"] = pd.Series(gap)
    out["log_lines_on_invoice"] = np.log1p(view.lines_on_invoice.astype(float))
    out.index = view.invoice_line_id.to_numpy()
    return out


def build_matrix(data):
    cfg = DetectionConfig()
    view = build_line_view(data, cfg)
    hist = price_history(view, cfg)
    X = pd.concat([build_features(view, hist, cfg), extra_history_features(view)], axis=1).astype(float)
    X.index.name = "invoice_line_id"
    assert_no_label_features(X)
    return view, X


# ---------------------------------------------------------------------------- models
class Prep:
    """inf->NaN, median imputation, RobustScaler, clipping - fitted on TRAIN only."""

    def fit(self, X):
        A = np.where(np.isfinite(X), X, np.nan)
        self.imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(A)
        self.sc = RobustScaler(quantile_range=(10, 90)).fit(self.imp.transform(A))
        self.sc.scale_[self.sc.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        A = np.where(np.isfinite(X), X, np.nan)
        return np.clip(self.sc.transform(self.imp.transform(A)), -CLIP, CLIP)


class IForest:
    name, family = "Isolation Forest", "tree"

    def fit(self, Z):
        self.m = IsolationForest(n_estimators=500, max_samples=256, random_state=SEED, n_jobs=1).fit(Z)
        return self

    def score(self, Z):
        return -self.m.score_samples(Z)


class LOF:
    name, family = "Local Outlier Factor", "distance"

    def fit(self, Z):
        self.m = LocalOutlierFactor(n_neighbors=35, novelty=True).fit(Z)
        return self

    def score(self, Z):
        return -self.m.score_samples(Z)


class ECOD:
    """Li et al. 2022: per-feature empirical tail probabilities from the training
    sample, skewness-aware; score = max(left-tail, right-tail, skew-chosen) sum of -log p."""
    name, family = "ECOD", "distribution"

    def fit(self, Z):
        self.S = np.sort(Z, axis=0)
        self.n = len(Z)
        mu, sd = Z.mean(0), Z.std(0) + 1e-12
        self.skew = ((Z - mu) ** 3).mean(0) / sd ** 3
        return self

    def _tails(self, Z):
        left = np.column_stack([np.searchsorted(self.S[:, j], Z[:, j], "right") for j in range(Z.shape[1])]) / self.n
        right = np.column_stack([self.n - np.searchsorted(self.S[:, j], Z[:, j], "left") for j in range(Z.shape[1])]) / self.n
        eps = 1.0 / (self.n + 1)
        return -np.log(np.clip(left, eps, 1)), -np.log(np.clip(right, eps, 1))

    def contributions(self, Z):
        L, R = self._tails(Z)
        return np.where(self.skew < 0, L, R), L, R

    def score(self, Z):
        A, L, R = self.contributions(Z)
        return np.maximum.reduce([L.sum(1), R.sum(1), A.sum(1)])


class RobustMahalanobis:
    name, family = "Robust Mahalanobis (MCD)", "linear / covariance"

    def fit(self, Z):
        Zj = Z + np.random.default_rng(SEED).normal(0, 1e-6, Z.shape)   # tiny jitter: binary columns
        self.m = MinCovDet(random_state=SEED, support_fraction=0.75).fit(Zj)
        return self

    def score(self, Z):
        return np.sqrt(self.m.mahalanobis(Z))


class Autoencoder:
    name, family = "Autoencoder (MLP 32-8-32)", "neural reconstruction"

    def fit(self, Z):
        self.m = MLPRegressor(hidden_layer_sizes=(32, 8, 32), activation="relu", alpha=1e-3, learning_rate_init=1e-3,
                              max_iter=800, early_stopping=True, validation_fraction=0.15, n_iter_no_change=25,
                              random_state=SEED).fit(Z, Z)
        return self

    def residuals(self, Z):
        return (Z - self.m.predict(Z)) ** 2

    def score(self, Z):
        return np.sqrt(self.residuals(Z).sum(1))


class Ensemble:
    name, family = "Consensus ensemble (IF + LOF + AE)", "rank average"

    def __init__(self, members):
        self.members = members

    def fit(self, Z):
        self.ref = [np.sort(m.score(Z)) for m in self.members]        # training-score ECDFs (calibration)
        return self

    def member_percentiles(self, Z):
        return np.column_stack([np.searchsorted(r, m.score(Z), "right") / len(r) for m, r in zip(self.members, self.ref)])

    def score(self, Z):
        return self.member_percentiles(Z).mean(1)


# ---------------------------------------------------------------------------- metrics
def fpr_at_recall(y, s, target):
    fpr, tpr, _ = roc_curve(y, s)
    i = np.searchsorted(tpr, target, "left")
    return float(fpr[min(i, len(fpr) - 1)])


def p_at_k(y, s, k):
    return float(y[np.argsort(-s, kind="mergesort")[:k]].mean())


def bootstrap_ci(y, s, fn, n=300, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].any() and not y[i].all():
            vals.append(fn(y[i], s[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------------------------------------------------------------------- plots
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # categorical slots 1-6, fixed order
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
POS, NEG = "#e34948", "#2a78d6"                                                 # diverging: toward / away from anomaly


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=10)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_curves(y, scores, metrics, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.6), dpi=200, facecolor=SURF)
    for (name, s), c in zip(scores.items(), COLORS):
        lw = 3.0 if name.startswith("Consensus") else 2.0
        f, t, _ = roc_curve(y, s)
        a1.plot(f, t, color=c, lw=lw, label=f"{name}  (AUC {metrics[name]['roc_auc']:.3f})")
        p, r, _ = precision_recall_curve(y, s)
        a2.plot(r, p, color=c, lw=lw, label=f"{name}  (AP {metrics[name]['pr_auc']:.3f})")
    a1.plot([0, 1], [0, 1], color=INK2, lw=1, ls=(0, (4, 4)), label="random ranking")
    a2.axhline(y.mean(), color=INK2, lw=1, ls=(0, (4, 4)), label=f"random ranking (base rate {y.mean():.1%})")
    for ax, (xl, yl, tt) in zip([a1, a2], [("False positive rate", "True positive rate (recall)", "ROC curve"),
                                           ("Recall", "Precision", "Precision-recall curve")]):
        _style(ax)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel(xl, color=INK2, fontsize=11); ax.set_ylabel(yl, color=INK2, fontsize=11)
        ax.set_title(tt, color=INK, fontsize=13, loc="left", fontweight="bold")
        ax.legend(frameon=False, fontsize=8.5, loc="lower right" if ax is a1 else "upper right", labelcolor=INK)
    fig.suptitle("Unsupervised detectors on the out-of-time window (fitted on earlier 60%, no labels)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def plot_waterfall(base, contribs, values, names, score, title, subtitle, path, xlabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = np.argsort(-np.abs(contribs))
    top = order[:8]
    rest = contribs[order[8:]].sum()
    vals = list(contribs[top]) + ([rest] if len(order) > 8 else [])
    labels = ([f"{names[i]} = {values[i]:.3g}   ({contribs[i]:+.2f})" for i in top]
              + ([f"{len(order) - 8} other features   ({rest:+.2f})"] if len(order) > 8 else []))
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=200, facecolor=SURF)
    _style(ax)
    ax.grid(True, axis="x", color=GRID); ax.grid(False, axis="y")
    cur = base
    ys = np.arange(len(vals))[::-1]
    for yv, v, lab in zip(ys, vals, labels):
        ax.barh(yv, v, left=cur, color=POS if v > 0 else NEG, height=0.62, edgecolor=SURF, linewidth=2)
        cur += v
    ax.set_yticks(ys); ax.set_yticklabels(labels, color=INK, fontsize=9.5)
    ax.axvline(base, color=INK2, lw=1, ls=(0, (4, 4)))
    ax.axvline(score, color=INK, lw=1.2)
    box = dict(boxstyle="round,pad=0.25", facecolor=SURF, edgecolor="none")
    ax.text(base, -0.95, f"average transaction\n{base:.2f}", color=INK2, fontsize=9, ha="center", va="center", bbox=box)
    ax.text(score, -0.95, f"this transaction\n{score:.2f}", color=INK, fontsize=9, ha="center", va="center", bbox=box)
    ax.set_xlim(min(base, score, base + min(np.cumsum(vals).min(), 0)) - 0.6, max(base, score, base + np.cumsum(vals).max()) + 0.9)
    ax.set_ylim(-1.5, len(vals) - 0.4)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=13, loc="left", fontweight="bold")
    fig.text(0.01, 0.005, subtitle, color=INK2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


# ---------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir")
    ap.add_argument("--out", default="reports_ml")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = load_procurement_data(a.data_dir)
    view, X = build_matrix(data)
    feat = list(X.columns)
    dates = pd.Series(view.invoice_date.to_numpy(), index=X.index)
    train = temporal_train_mask(dates, TRAIN_FRACTION)
    prep = Prep().fit(X.values[train])
    Ztr, Zte = prep.transform(X.values[train]), prep.transform(X.values[~train])
    print(f"features {len(feat)} | train {train.sum()} lines (to {dates[train].max().date()}) | "
          f"test {(~train).sum()} lines (from {dates[~train].min().date()})")

    iforest, lof, ae = IForest(), LOF(), Autoencoder()
    models = [iforest, lof, ECOD(), RobustMahalanobis(), ae]
    fit_s, test_scores, train_scores, lat = {}, {}, {}, {}
    for m in models:
        t = time.perf_counter(); m.fit(Ztr); fit_s[m.name] = time.perf_counter() - t
    ens = Ensemble([iforest, lof, ae])
    t = time.perf_counter(); ens.fit(Ztr); fit_s[ens.name] = time.perf_counter() - t + sum(fit_s[x.name] for x in ens.members)
    models.append(ens)
    for m in models:
        train_scores[m.name] = m.score(Ztr)
        reps = []
        for _ in range(5):
            t = time.perf_counter(); s = m.score(Zte); reps.append(time.perf_counter() - t)
        test_scores[m.name] = s
        lat[m.name] = 1000 * float(np.median(reps)) / len(Zte) * 1000      # ms per 1,000 transactions

    # ---- labels: loaded only now, after all scoring
    from leak_hunter.evaluation import load_ground_truth
    gt = load_ground_truth(a.data_dir).set_index("invoice_line_id")
    test_ids = X.index[~train]
    y = (gt.loc[test_ids, "is_anomaly"].to_numpy() == 1)
    ptype = gt.loc[test_ids, "primary_anomaly_type"].to_numpy()

    M = {}
    for m in models:
        s = test_scores[m.name]
        thr = float(np.quantile(train_scores[m.name], 1 - REVIEW_BUDGET))
        flag = s >= thr
        M[m.name] = dict(
            family=m.family,
            p_at_10=p_at_k(y, s, 10), p_at_25=p_at_k(y, s, 25), p_at_50=p_at_k(y, s, 50),
            roc_auc=float(roc_auc_score(y, s)), pr_auc=float(average_precision_score(y, s)),
            pr_auc_ci95=bootstrap_ci(y, s, average_precision_score),
            fpr_at_recall_50=fpr_at_recall(y, s, 0.50), fpr_at_recall_80=fpr_at_recall(y, s, 0.80),
            review_flagged=int(flag.sum()), review_precision=float(y[flag].mean()) if flag.any() else 0.0,
            review_recall=float(y[flag].sum() / y.sum()),
            recall_by_type_at_review={t: f"{int((flag & (ptype == t)).sum())}/{int((ptype == t).sum())}"
                                      for t in ["PO_PRICE_MISMATCH", "QUANTITY_DISCREPANCY",
                                                "HISTORICAL_PRICE_SPIKE", "DUPLICATE_INVOICE"]},
            latency_ms_per_1000=lat[m.name], fit_seconds=fit_s[m.name])
        print(f"{m.name:<36} P@10 {M[m.name]['p_at_10']:.2f} P@25 {M[m.name]['p_at_25']:.2f} P@50 {M[m.name]['p_at_50']:.2f} "
              f"ROC {M[m.name]['roc_auc']:.3f} AP {M[m.name]['pr_auc']:.3f} FPR@R80 {M[m.name]['fpr_at_recall_80']:.3f} "
              f"lat {lat[m.name]:.1f}ms/1k")
    winner = max(M, key=lambda k: M[k]["pr_auc"])

    # ---- attribution for the #1 transaction of the winner
    import shap
    s_w = test_scores[winner]
    top_i = int(np.argmax(s_w))
    top_id = test_ids[top_i]
    wm = next(m for m in models if m.name == winner)
    rng = np.random.default_rng(SEED)
    bg = Ztr[rng.choice(len(Ztr), 100, replace=False)]
    if isinstance(wm, IForest):
        ex = shap.TreeExplainer(wm.m)
        sv = ex.shap_values(Zte[top_i:top_i + 1])[0] * -1          # shap explains score_samples; flip to anomaly direction
        base = float(-ex.expected_value) if np.ndim(ex.expected_value) == 0 else float(-ex.expected_value[0])
        method = "TreeSHAP on the Isolation Forest"
    else:
        f = lambda Z: wm.score(np.asarray(Z, dtype=float))
        ex = shap.KernelExplainer(f, shap.kmeans(Ztr, 25))
        sv = ex.shap_values(Zte[top_i:top_i + 1], nsamples=1500, silent=True)[0]
        base = float(ex.expected_value)
        method = "Kernel SHAP on the model score function (k-means background of 25 training points)"
    raw = X.loc[top_id].to_numpy()
    final = base + float(np.sum(sv))          # additive: expected value + contributions = explained model output
    unit = ("negative mean isolation depth (higher = isolated in fewer splits = more anomalous)"
            if isinstance(wm, IForest) else f"{winner} score")
    plot_waterfall(base, np.asarray(sv), raw, feat, final,
                   f"Why {top_id} is ranked #1 by the {winner}",
                   f"{method}. Red pushes toward 'anomalous', blue toward 'normal'. Values are raw (unscaled) feature values. "
                   f"Ground truth: {gt.loc[top_id, 'primary_anomaly_type']}.",
                   out / "top1_attribution_waterfall.png", unit)
    # reconstruction-error breakdown (autoencoder) for the same transaction, for auditors
    rec = ae.residuals(Zte[top_i:top_i + 1])[0]
    rec_share = sorted(zip(feat, rec / rec.sum()), key=lambda t: -t[1])[:5]

    # top-25 table with per-model percentile ranks and top drivers
    pct = ens.member_percentiles(Zte)
    order = np.argsort(-s_w)[:25]
    top = pd.DataFrame({"rank": np.arange(1, 26), "invoice_line_id": test_ids[order],
                        "winner_score": np.round(s_w[order], 4),
                        "IF_pctl": pct[order, 0].round(3), "LOF_pctl": pct[order, 1].round(3), "AE_pctl": pct[order, 2].round(3),
                        "ground_truth": [gt.loc[i, "primary_anomaly_type"] for i in test_ids[order]]})
    top.to_csv(out / "top_anomalies.csv", index=False)

    # rules overlap: does any detector find anomalies the rule engine misses?
    plot_curves(y, test_scores, M, out / "roc_pr_curves.png")
    summary = dict(features=feat, n_train=int(train.sum()), n_test=int((~train).sum()), base_rate_test=float(y.mean()),
                   winner=winner, top1=dict(invoice_line_id=top_id, ground_truth=gt.loc[top_id, "primary_anomaly_type"],
                                            shap=dict(zip(feat, map(float, sv))), shap_base=base, shap_output=final,
                                            shap_units=unit,
                                            autoencoder_error_share=dict(rec_share)),
                   metrics=M, runtime_seconds=round(time.time() - t0, 1))
    (out / "benchmark.json").write_text(json.dumps(summary, indent=2, default=str))

    L = ["# Unsupervised anomaly detection benchmark - AI Leak Hunter", "",
         f"Out-of-time evaluation: fitted on the earliest {TRAIN_FRACTION:.0%} of invoice lines ({train.sum():,}), "
         f"scored on the remaining {(~train).sum():,} (planted-anomaly base rate {y.mean():.1%}). {len(feat)} leak-free "
         "features; no labels used for fitting, tuning, weighting or thresholds.", "",
         "| # | Model | Family | P@10 | P@25 | P@50 | ROC-AUC | PR-AUC (95% CI) | FPR @ 50% recall | FPR @ 80% recall | Review-budget precision / recall | Latency (ms / 1,000 txns) | Fit (s) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tags = ["1", "2a", "2b", "3", "4", "5"]
    for tg, (n, d) in zip(tags, M.items()):
        bold = "**" if n == winner else ""
        L.append(f"| {tg} | {bold}{n}{bold} | {d['family']} | {d['p_at_10']:.2f} | {d['p_at_25']:.2f} | {d['p_at_50']:.2f} | "
                 f"{d['roc_auc']:.3f} | {bold}{d['pr_auc']:.3f}{bold} ({d['pr_auc_ci95'][0]:.3f}-{d['pr_auc_ci95'][1]:.3f}) | "
                 f"{d['fpr_at_recall_50']:.3f} | {d['fpr_at_recall_80']:.3f} | {d['review_precision']:.2f} / {d['review_recall']:.2f} "
                 f"({d['review_flagged']} flagged) | {d['latency_ms_per_1000']:.1f} | {d['fit_seconds']:.1f} |")
    L += ["", "Review budget = lines scoring above the 98th percentile of training scores.", "",
          "Planted anomalies flagged at the review budget, by type:", "",
          "| Model | Price mismatch | Quantity | Price spike | Duplicate |", "|---|---|---|---|---|"]
    for n, d in M.items():
        r = d["recall_by_type_at_review"]
        L.append(f"| {n} | {r['PO_PRICE_MISMATCH']} | {r['QUANTITY_DISCREPANCY']} | {r['HISTORICAL_PRICE_SPIKE']} | {r['DUPLICATE_INVOICE']} |")
    L += ["", f"#1 transaction of the winner ({winner}): `{top_id}` - ground truth "
          f"{gt.loc[top_id, 'primary_anomaly_type']}. Top attributions: " +
          ", ".join(f"{feat[i]} {sv[i]:+.3f}" for i in np.argsort(-np.abs(sv))[:4]) +
          ". Autoencoder error share: " + ", ".join(f"{k} {v:.0%}" for k, v in rec_share[:3]) + ".", ""]
    (out / "benchmark.md").write_text("\n".join(L) + "\n")
    print(f"winner by PR-AUC: {winner}; outputs in {out.resolve()} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
