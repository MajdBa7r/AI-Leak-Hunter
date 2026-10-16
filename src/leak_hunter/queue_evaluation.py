"""Investigation-queue evaluation: rules-only vs ML-only vs integrated.

What an investigator experiences is a QUEUE: review cases top-down until time runs
out. This module measures each ordering on the out-of-time window (invoice lines
after the Isolation Forest training cutoff) at fixed review budgets K:

  precision@K          share of the first K cases that contain a planted anomaly
  exposure captured@K  ground-truth expected exposure inside the first K cases /
                       all ground-truth exposure in the window
  cases to 50% / 80%   cases that must be reviewed to capture that share of exposure
  queue precision / line recall  over the whole queue

Orderings (no ordering uses labels; risk-score weights were fixed beforehand):
  RULES_SEVERITY     rule incidents by severity label, then exposure (previous default)
  RULES_EXPOSURE     rule incidents by de-duplicated potential exposure
  ML_ONLY            every invoice line by Isolation Forest score
  INTEGRATED         rule + Isolation-Forest-only incidents by risk score
  INTEGRATED_NO_ML   same risk score without the anomaly points (ablation)

Scenarios: clean data, each Phase 1 disruption alone, and all seven combined.

    PYTHONPATH=src python -m leak_hunter.queue_evaluation [--out reports_queue]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import load_procurement_data
from .evaluation import load_ground_truth
from .pipeline import run_detection
from .scenarios import SCENARIOS, apply

KS = [25, 50, 100, 200]
ORDERINGS = ["RULES_SEVERITY", "RULES_EXPOSURE", "ML_ONLY", "INTEGRATED", "INTEGRATED_NO_ML", "EXPECTED_VALUE"]
LABELS = {"RULES_SEVERITY": "Rules, by severity label (previous default)",
          "RULES_EXPOSURE": "Rules, by exposure",
          "ML_ONLY": "ML only (Isolation Forest score)",
          "INTEGRATED": "Integrated risk score (rules + ML)",
          "INTEGRATED_NO_ML": "Integrated risk score without ML points",
          "EXPECTED_VALUE": "Expected value (exposure x confidence), added after v1 results"}


def build_queues(res) -> tuple[dict, set]:
    """Ordered lists of cases (each a list of invoice_line_ids) restricted to the test window."""
    s = res.scores
    test_lines = set(s.invoice_line_id[~s.in_training_window])
    inc = res.incidents.copy()
    inc["lines"] = inc.invoice_line_ids.str.split("|")
    inc = inc[inc.lines.map(lambda ls: any(l in test_lines for l in ls))]
    rules = inc[inc.detection_methods.str.contains("RULE_BASED")]
    q = {
        "RULES_SEVERITY": rules.sort_values(["severity_rank"]).lines.tolist(),
        "RULES_EXPOSURE": rules.sort_values(["potential_exposure", "incident_id"], ascending=[False, True]).lines.tolist(),
        "ML_ONLY": [[l] for l in s[~s.in_training_window].sort_values("anomaly_score", ascending=False).invoice_line_id],
        "INTEGRATED": inc.sort_values(["risk_score", "potential_exposure", "incident_id"],
                                      ascending=[False, False, True]).lines.tolist(),
        "INTEGRATED_NO_ML": inc.assign(r=inc.risk_score - inc.pts_anomaly).sort_values(
            ["r", "potential_exposure", "incident_id"], ascending=[False, False, True]).lines.tolist(),
        "EXPECTED_VALUE": inc.sort_values(["value_rank"]).lines.tolist(),
    }
    return q, test_lines


def queue_metrics(cases: list[list[str]], gt: pd.DataFrame, test_lines: set, ks=KS) -> dict:
    y = gt.is_anomaly.eq(1)
    ex = gt.expected_exposure.where(y, 0.0)
    test_anom = [l for l in test_lines if y.get(l, False)]
    total_exp = float(ex.reindex(list(test_lines)).sum())
    hit = np.array([any(y.get(l, False) for l in c) for c in cases], dtype=bool)
    seen, cum = set(), []
    for c in cases:
        seen.update(l for l in c if l in test_lines)
        cum.append(float(ex.reindex(list(seen)).sum()) if seen else 0.0)
    cum = np.array(cum)
    covered = {l for c in cases for l in c if l in test_lines and y.get(l, False)}

    def need(share):
        idx = np.nonzero(cum >= share * total_exp - 1e-9)[0]
        return int(idx[0] + 1) if len(idx) else None

    out = {"queue_length": len(cases), "queue_precision": round(float(hit.mean()), 4) if len(hit) else 0.0,
           "line_recall": round(len(covered) / max(1, len(test_anom)), 4),
           "cases_for_50pct_exposure": need(0.5), "cases_for_80pct_exposure": need(0.8),
           "total_test_exposure": round(total_exp, 2)}
    for k in ks:
        kk = min(k, len(cases))
        out[f"precision_at_{k}"] = round(float(hit[:kk].mean()), 4) if kk else 0.0
        out[f"exposure_captured_at_{k}"] = round(float(cum[kk - 1] / total_exp), 4) if kk and total_exp else 0.0
    out["_curve"] = (cum / total_exp).round(4).tolist() if total_exp else []
    return out


def evaluate_scenario(data, gt_idx) -> dict:
    res = run_detection(data=data)
    queues, test_lines = build_queues(res)
    return {o: queue_metrics(queues[o], gt_idx, test_lines) for o in ORDERINGS}


COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def plot_capture(results: dict, scenarios: list[str], path: Path, max_cases: int = 300):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6.5 * len(scenarios), 5.4), dpi=200, facecolor=SURF,
                             sharey=True)
    for ax, sc in zip(np.atleast_1d(axes), scenarios):
        for o, c in zip(ORDERINGS, COLORS):
            curve = results[sc][o]["_curve"][:max_cases]
            lw = 3.0 if o in ("INTEGRATED", "EXPECTED_VALUE") else 2.0
            ax.plot(np.arange(1, len(curve) + 1), curve, color=c, lw=lw,
                    label=f"{LABELS[o]}  (@50: {results[sc][o]['exposure_captured_at_50']:.0%})")
        ax.set_facecolor(SURF)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color(GRID)
        ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=10)
        ax.set_xlim(0, max_cases); ax.set_ylim(0, 1.02)
        ax.axvline(50, color=INK2, lw=1, ls=(0, (4, 4)))
        ax.set_title(sc, color=INK, fontsize=12, loc="left", fontweight="bold")
        ax.set_xlabel("Cases reviewed (top of queue)", color=INK2)
        ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    np.atleast_1d(axes)[0].set_ylabel("Share of true exposure captured", color=INK2)
    fig.suptitle("How fast each investigation queue reaches the money (out-of-time window)", color=INK,
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def render(results: dict) -> str:
    L = ["# Investigation-queue evaluation", "",
         "Unit = case (an incident; for ML-only, a single invoice line). Window = invoice lines after the Isolation "
         "Forest training cutoff (out-of-time). Labels are used only to score the queues. Risk-score weights were "
         "fixed in `config.PrioritizationConfig` before this evaluation was first run and have not been changed "
         "since. Single run per scenario (seeded); no confidence intervals.", ""]
    for sc, r in results.items():
        L += [f"## {sc}", "", "| Ordering | Queue | P@25 | P@50 | P@100 | Exposure @25 | @50 | @100 | @200 | "
              "Cases to 80% of exposure | Queue precision | Line recall |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for o in ORDERINGS:
            m = r[o]
            L.append(f"| {LABELS[o]} | {m['queue_length']} | {m['precision_at_25']:.2f} | {m['precision_at_50']:.2f} | "
                     f"{m['precision_at_100']:.2f} | {m['exposure_captured_at_25']:.0%} | {m['exposure_captured_at_50']:.0%} | "
                     f"{m['exposure_captured_at_100']:.0%} | {m['exposure_captured_at_200']:.0%} | "
                     f"{m['cases_for_80pct_exposure'] or 'not reached'} | {m['queue_precision']:.2f} | {m['line_recall']:.2f} |")
        L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir")
    ap.add_argument("--out", default="reports_queue")
    ap.add_argument("--quick", action="store_true", help="clean + combined only")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    base = load_procurement_data(a.data_dir)
    gt = load_ground_truth(a.data_dir)
    gt_idx = gt.set_index("invoice_line_id")
    names = [n for n, _ in SCENARIOS] if not a.quick else ["Clean data (no disruption)", "ALL SEVEN COMBINED"]
    results = {}
    for name in names:
        t = time.time()
        fns = dict(SCENARIOS)[name]
        data = apply(base, gt, [f.__name__ for f in fns]) if fns else apply(base, gt, [])
        results[name] = evaluate_scenario(data, gt_idx)
        r = results[name]
        print(f"{name:<58} " + " | ".join(f"{o}: P@50 {r[o]['precision_at_50']:.2f} Exp@50 {r[o]['exposure_captured_at_50']:.0%}"
                                          for o in ["RULES_EXPOSURE", "ML_ONLY", "INTEGRATED", "EXPECTED_VALUE"]) +
              f"  ({time.time() - t:.0f}s)")
    (out / "queue_evaluation.md").write_text(render(results))
    (out / "queue_evaluation.json").write_text(json.dumps(
        {s: {o: {k: v for k, v in m.items() if k != "_curve"} for o, m in r.items()} for s, r in results.items()}, indent=2))
    plot_capture(results, ["Clean data (no disruption)", "ALL SEVEN COMBINED"], out / "exposure_capture_curves.png")
    print(f"written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
