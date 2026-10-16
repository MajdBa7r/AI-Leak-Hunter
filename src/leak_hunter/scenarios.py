"""Realistic operational disruptions for robustness and queue evaluation.

Each function edits a copy of ProcurementData in place, simulating one
real-world complication at the intensity fixed in Phase 1 (before any target
metric existed). Ground truth is never changed: a disruption that makes a
normal line look anomalous yields a false positive, one that hides a planted
anomaly yields a false negative. ``gt`` is used only to choose WHICH records a
disruption touches (e.g. re-entered duplicates), never by any detector.

Used by scripts/rules_noise_benchmark.py and leak_hunter.queue_evaluation.
"""
from __future__ import annotations

import copy
import re

import numpy as np
import pandas as pd

SEED = 7


def fuzz_numbers(d, gt, rng):
    kinds = ["suffix", "swap", "O_for_0", "drop_zero", "add_zero", "case_spacing"]
    reps = gt.loc[gt.scenario_tags.str.contains("DUP_REPEAT_ENTRY"), "invoice_id"].tolist()
    taken = set(zip(d.invoices.supplier_id, d.invoices.invoice_number))

    def f(s, k):
        if k == "suffix":
            return s + "-A"
        if k == "swap":
            ds = [i for i, c in enumerate(s) if c.isdigit()]
            for a in range(len(ds) - 1, 0, -1):
                i, j = ds[a - 1], ds[a]
                if j == i + 1 and s[i] != s[j]:
                    return s[:i] + s[j] + s[i] + s[j + 1:]
            return s + "-A"
        if k == "O_for_0":
            return s.replace("0", "O", 1) if "0" in s else s + "-A"
        if k == "drop_zero":
            o = re.sub(r"(?<![0-9])0(?=[0-9])", "", s, count=1)
            return o if o != s else s + "-A"
        if k == "add_zero":
            return re.sub(r"(\d+)$", lambda m: "0" + m.group(1), s)
        return " " + s.lower().replace("-", " ") + " "
    for inv in reps:
        m = d.invoices.invoice_id == inv
        sup, num = d.invoices.loc[m, "supplier_id"].iloc[0], d.invoices.loc[m, "invoice_number"].iloc[0]
        new = f(num, kinds[int(rng.integers(len(kinds)))])
        if (sup, new) in taken:
            new = num + "-A"
        d.invoices.loc[m, "invoice_number"] = new


def clone_vendor(d, gt, rng):
    reps = gt.loc[gt.scenario_tags.str.contains("DUP_REPEAT_ENTRY"), "invoice_id"]
    pick = reps.sample(frac=0.25, random_state=SEED)          # a quarter of re-entries land on a duplicate vendor record
    m = d.invoices.invoice_id.isin(pick)
    d.invoices.loc[m, "supplier_id"] = d.invoices.loc[m, "supplier_id"] + "-B"


def posting_lag(d, gt, rng, days=21):
    as_of = max(d.invoices.invoice_date.max(), d.goods_receipts.receipt_date.max())
    d.goods_receipts = d.goods_receipts[d.goods_receipts.receipt_date <= as_of - pd.Timedelta(days=days)].reset_index(drop=True)


def price_noise(d, gt, rng):
    normal = d.invoices.invoice_line_id.isin(set(gt.invoice_line_id[gt.is_anomaly == 0]))
    m = normal & (rng.random(len(d.invoices)) < 0.10)
    d.invoices.loc[m, "unit_price"] = (d.invoices.loc[m, "unit_price"]
                                       + rng.choice([-0.01, 0.01, 0.004, 0.006], int(m.sum()))).round(3)


def retro_amendments(d, gt, rng):
    am = d.po_amendments
    first = d.invoices.groupby("po_line_id").invoice_date.min()
    m = (am.status == "Approved") & (am.amendment_type == "PRICE")
    am.loc[m, "amendment_date"] = am.loc[m, "po_line_id"].map(first) + pd.Timedelta(days=5)


def renegotiation(d, gt, rng):
    normal = set(gt.invoice_line_id[gt.is_anomaly == 0])
    sups = d.suppliers.supplier_id.sample(3, random_state=1).tolist()
    po = d.purchase_orders
    pm = po.supplier_id.isin(sups) & (po.po_date >= "2026-04-01")
    po.loc[pm, "unit_price"] = (po.loc[pm, "unit_price"] * 1.35).round(2)
    raised = po.po_line_id[pm]
    im = d.invoices.po_line_id.isin(raised) & d.invoices.invoice_line_id.isin(normal)
    d.invoices.loc[im, "unit_price"] = (d.invoices.loc[im, "unit_price"] * 1.35).round(2)
    am = d.po_amendments
    mm = am.po_line_id.isin(raised) & am.new_unit_price.notna()
    am.loc[mm, ["previous_unit_price", "new_unit_price"]] = (am.loc[mm, ["previous_unit_price", "new_unit_price"]] * 1.35).round(2)


def returns(d, gt, rng):
    g = d.goods_receipts
    pick = g.sample(frac=0.05, random_state=3)
    ret = pick.assign(receipt_id=pick.receipt_id + "-RET",
                      quantity_received=-(pick.quantity_received * 0.2).round().clip(lower=1),
                      receipt_date=pick.receipt_date + pd.Timedelta(days=10))
    d.goods_receipts = pd.concat([g, ret], ignore_index=True)


SCENARIOS = [
    ("Clean data (no disruption)", []),
    ("Fuzzy duplicate invoice numbers (all 40 re-entries)", [fuzz_numbers]),
    ("25% of re-entries booked under a duplicate vendor record", [clone_vendor]),
    ("Warehouse posting lag 21 days", [posting_lag]),
    ("Rounding noise (+/-1 cent, 3 decimals) on 10% of lines", [price_noise]),
    ("Price amendments recorded 5 days after first invoice", [retro_amendments]),
    ("Legitimate 35% renegotiation (3 suppliers)", [renegotiation]),
    ("Returns posted as negative receipts (5% of receipts)", [returns]),
    ("ALL SEVEN COMBINED", [fuzz_numbers, clone_vendor, posting_lag, price_noise, retro_amendments,
                            renegotiation, returns]),
]




def apply(data, gt, names: list[str] | None = None, seed: int = SEED):
    """Return a disrupted deep copy of ``data`` (all seven when names is None)."""
    d, rng = copy.deepcopy(data), np.random.default_rng(seed)
    fns = dict(SCENARIOS)["ALL SEVEN COMBINED"] if names is None else [ALL[n] for n in names]
    for fn in fns:
        fn(d, gt, rng)
    return d


ALL = {f.__name__: f for f in [fuzz_numbers, clone_vendor, posting_lag, price_noise, retro_amendments,
                               renegotiation, returns]}
