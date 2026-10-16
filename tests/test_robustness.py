"""Robustness (stress) suite: realistic disruptions of the repository dataset.

Each test perturbs a copy of the real data in one realistic way and checks the
expected behaviour of the rule engine (Isolation Forest disabled for speed).
Two disruptions need schema the dataset does not have yet and are marked
``xfail(strict=True)``: they document known gaps and will start failing
loudly (XPASS) once Phase 2 closes them, so the marker must then be removed.

Disruption families
1. fuzzy duplicate invoice numbers   -> must still pair all 40 repeat entries
2. duplicate vendor master record    -> KNOWN GAP (needs tax ID / bank account)
3. warehouse posting lag             -> grace period absorbs lag shorter than it
4. 1-cent / 3-decimal price noise    -> default tolerance absorbs it
5. retroactive price amendments      -> KNOWN GAP (needs amendment effective_date)
6. legitimate contract renegotiation -> only zero-exposure, low-severity spike flags
7. goods returns (negative receipts) -> flags confined to returned quantities
plus metamorphic checks: row order and later transactions must not change results.
"""
from __future__ import annotations

import copy
import re
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from leak_hunter.config import DetectionConfig
from leak_hunter.data_loader import load_procurement_data
from leak_hunter.pipeline import run_detection

RULES_ONLY = replace(DetectionConfig(), isolation_forest=replace(DetectionConfig().isolation_forest, enabled=False))


@pytest.fixture(scope="module")
def base(real_data_dir):
    return load_procurement_data(real_data_dir)


@pytest.fixture(scope="module")
def gt(ground_truth):
    return ground_truth


@pytest.fixture(scope="module")
def baseline(base):
    return run_detection(data=copy.deepcopy(base), config=RULES_ONLY)


def detect(data, config=RULES_ONLY):
    res = run_detection(data=data, config=config)
    return res, res.findings[res.findings.detection_method == "RULE_BASED"]


def outcome(rb, res, gt):
    pred = set(rb.invoice_line_id)
    anomalies = set(gt.invoice_line_id[gt.is_anomaly == 1])
    pending = set(res.pending.invoice_line_id)
    return {"fp": pred - anomalies, "missed": anomalies - pred - pending, "pending": pending}


def repeat_docs(gt):
    return gt.loc[gt.scenario_tags.str.contains("DUP_REPEAT_ENTRY"), "invoice_id"].tolist()


# ------------------------------------------------------------------ baseline
def test_clean_baseline(baseline, gt):
    rb = baseline.findings[baseline.findings.detection_method == "RULE_BASED"]
    o = outcome(rb, baseline, gt)
    assert not o["fp"] and not o["missed"]
    assert len(o["pending"]) <= 5           # only invoices younger than the grace period


# ------------------------------------------------------------------ 1. fuzzy duplicates
def _fuzz(s: str, kind: str) -> str:
    if kind == "case_and_spacing":
        return " " + s.lower().replace("-", " ") + " "
    if kind == "suffix":
        return s + "-A"
    if kind == "adjacent_swap":
        digits = [i for i, c in enumerate(s) if c.isdigit()]
        for k in range(len(digits) - 1, 0, -1):
            i, j = digits[k - 1], digits[k]
            if j == i + 1 and s[i] != s[j]:
                return s[:i] + s[j] + s[i] + s[j + 1:]
        return s + "-A"
    if kind == "letter_O_for_zero":
        return s.replace("0", "O", 1)
    if kind == "drop_leading_zero":
        out = re.sub(r"(?<![0-9])0(?=[0-9])", "", s, count=1)
        return out if out != s else s + "-A"
    if kind == "add_leading_zero":
        return re.sub(r"(\d+)$", lambda m: "0" + m.group(1), s)
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["case_and_spacing", "suffix", "adjacent_swap", "letter_O_for_zero",
                                  "drop_leading_zero", "add_leading_zero"])
def test_fuzzy_duplicate_numbers_still_paired(base, gt, kind):
    d = copy.deepcopy(base)
    reps = repeat_docs(gt)
    m = d.invoices.invoice_id.isin(reps)
    d.invoices.loc[m, "invoice_number"] = d.invoices.loc[m, "invoice_number"].map(lambda s: _fuzz(s, kind))
    # keep the disruption honest: an altered number must not coincide with ANOTHER real invoice of the
    # same supplier (that would be an exact number reuse, a different scenario that the exact rule links)
    taken = set(zip(base.invoices.supplier_id, base.invoices.invoice_number))
    clash = m & pd.Series([(a, b) in taken for a, b in zip(d.invoices.supplier_id, d.invoices.invoice_number)],
                          index=d.invoices.index)
    d.invoices.loc[clash, "invoice_number"] = base.invoices.loc[clash, "invoice_number"].map(lambda s: _fuzz(s, "suffix"))
    assert (d.invoices.loc[m, "invoice_number"] != base.invoices.loc[m, "invoice_number"]).all()
    res, rb = detect(d)
    dup = rb[rb.finding_type == "DUPLICATE_INVOICE"]
    pairs = gt[gt.duplicate_group_id != ""].groupby("duplicate_group_id").invoice_line_id.apply(set)
    inc_lines = set(map(frozenset, dup.groupby("incident_id").invoice_line_id.apply(set)))
    dates = d.invoices.set_index("invoice_line_id").invoice_date
    gap = pairs.map(lambda s_: abs((dates[max(s_)] - dates[min(s_)]).days))
    interval = RULES_ONLY.rules.dup_recurring_interval_days
    found = pairs.map(lambda s_: frozenset(s_) in inc_lines)
    if kind == "case_and_spacing":           # still an exact match: the recurring exemption does not apply
        assert found.all()
    else:                                    # every fuzzy pair is found, except the documented trade-off:
        assert found[gap != interval].all()  # re-entries exactly `interval` days later look like recurring billing
        assert not found[gap == interval].any()
    assert not outcome(rb, res, gt)["fp"]


# ------------------------------------------------------------------ 2. vendor master duplicate
@pytest.mark.xfail(strict=True, reason="KNOWN GAP: a second vendor record for the same supplier hides the duplicate. "
                                       "Needs supplier tax ID / bank account in the schema (Phase 2).")
def test_duplicate_under_cloned_vendor_record_is_detected(base, gt):
    d = copy.deepcopy(base)
    m = d.invoices.invoice_id.isin(repeat_docs(gt))
    d.invoices.loc[m, "supplier_id"] = d.invoices.loc[m, "supplier_id"] + "-B"
    _, rb = detect(d)
    assert (rb.finding_type == "DUPLICATE_INVOICE").sum() == 80


def test_cloned_vendor_duplicates_still_caught_by_quantity_rule_when_same_po_line(base, gt):
    """Partial safety net while the gap is open: the repeat over-bills its PO line."""
    d = copy.deepcopy(base)
    reps = repeat_docs(gt)
    m = d.invoices.invoice_id.isin(reps)
    d.invoices.loc[m, "supplier_id"] = d.invoices.loc[m, "supplier_id"] + "-B"
    _, rb = detect(d)
    rep_lines = set(d.invoices.invoice_line_id[m])
    assert rep_lines <= set(rb.invoice_line_id[rb.finding_type == "QUANTITY_DISCREPANCY"])


# ------------------------------------------------------------------ 3. posting lag
def _drop_late_receipts(d, lag_days):
    as_of = max(d.invoices.invoice_date.max(), d.goods_receipts.receipt_date.max())
    late = d.goods_receipts.receipt_date > as_of - pd.Timedelta(days=lag_days)
    lost = set(d.goods_receipts.po_line_id[late])
    d.goods_receipts = d.goods_receipts[~late].reset_index(drop=True)
    return lost


def test_posting_lag_shorter_than_grace_is_mostly_absorbed(base, gt):
    d0, d1 = copy.deepcopy(base), copy.deepcopy(base)
    lost = _drop_late_receipts(d0, 21)
    _drop_late_receipts(d1, 21)
    no_grace = replace(RULES_ONLY, rules=replace(RULES_ONLY.rules, qty_grace_days=0))
    res0, rb0 = detect(d0, no_grace)
    res1, rb1 = detect(d1)
    fp0, fp1 = outcome(rb0, res0, gt)["fp"], outcome(rb1, res1, gt)["fp"]
    assert len(fp0) > 0
    assert len(fp1) <= 0.25 * len(fp0)                       # v1: 35 -> hardened: 7
    fp_pl = set(rb1[rb1.invoice_line_id.isin(fp1)].po_line_id)
    assert fp_pl <= lost                                     # every residual alarm is on a PO line that lost a receipt
    assert len(res1.pending) > len(res0.pending)


def test_posting_lag_longer_than_grace_stays_confined(base, gt):
    d = copy.deepcopy(base)
    lost = _drop_late_receipts(d, 60)
    res, rb = detect(d)
    o = outcome(rb, res, gt)
    assert not o["missed"]
    assert set(rb[rb.invoice_line_id.isin(o["fp"])].po_line_id) <= lost   # no collateral false alarms


# ------------------------------------------------------------------ 4. price rounding noise
def _price_noise(d, gt, seed=7):
    rng = np.random.default_rng(seed)
    normal = d.invoices.invoice_line_id.isin(set(gt.invoice_line_id[gt.is_anomaly == 0]))
    m = normal & (rng.random(len(d.invoices)) < 0.10)
    d.invoices.loc[m, "unit_price"] = (d.invoices.loc[m, "unit_price"]
                                       + rng.choice([-0.01, 0.01, 0.004, 0.006], int(m.sum()))).round(3)
    return int(m.sum())


def test_rounding_noise_absorbed_by_default_tolerance(base, gt):
    d = copy.deepcopy(base)
    assert _price_noise(d, gt) > 300
    res, rb = detect(d)
    o = outcome(rb, res, gt)
    assert not o["fp"] and not o["missed"]


def test_rounding_noise_would_flag_with_zero_tolerance(base, gt):
    d = copy.deepcopy(base)
    _price_noise(d, gt)
    zero = replace(RULES_ONLY, rules=replace(RULES_ONLY.rules, price_tolerance_abs=0.0, price_tolerance_pct=0.0,
                                             price_materiality_abs=0.0, price_materiality_pct=0.0))
    res, rb = detect(d, zero)
    assert len(outcome(rb, res, gt)["fp"]) > 100            # proves the disruption is real (v1 behaviour)


# ------------------------------------------------------------------ 5. retroactive amendments
@pytest.mark.xfail(strict=True, reason="KNOWN GAP: an amendment approved after the invoice but effective before it "
                                       "cannot be expressed - po_amendments has one date. Needs effective_date (Phase 2).")
def test_retroactive_price_amendments_are_honoured(base, gt):
    d = copy.deepcopy(base)
    am = d.po_amendments
    first_inv = d.invoices.groupby("po_line_id").invoice_date.min()
    m = (am.status == "Approved") & (am.amendment_type == "PRICE")
    am.loc[m, "amendment_date"] = am.loc[m, "po_line_id"].map(first_inv) + pd.Timedelta(days=5)
    res, rb = detect(d)
    assert not outcome(rb, res, gt)["fp"]


# ------------------------------------------------------------------ 6. contract renegotiation
def test_legitimate_renegotiation_raises_only_zero_exposure_spike_flags(base, gt):
    d = copy.deepcopy(base)
    normal = set(gt.invoice_line_id[gt.is_anomaly == 0])
    sups = d.suppliers.supplier_id.sample(3, random_state=1).tolist()
    po = d.purchase_orders
    pm = po.supplier_id.isin(sups) & (po.po_date >= "2026-04-01")     # new contract prices on new POs ...
    po.loc[pm, "unit_price"] = (po.loc[pm, "unit_price"] * 1.35).round(2)
    raised = po.po_line_id[pm]
    im = d.invoices.po_line_id.isin(raised) & d.invoices.invoice_line_id.isin(normal)   # ... billed as agreed
    d.invoices.loc[im, "unit_price"] = (d.invoices.loc[im, "unit_price"] * 1.35).round(2)
    am = d.po_amendments
    mm = am.po_line_id.isin(raised) & am.new_unit_price.notna()
    am.loc[mm, ["previous_unit_price", "new_unit_price"]] = (am.loc[mm, ["previous_unit_price", "new_unit_price"]] * 1.35).round(2)
    res, rb = detect(d)
    fp = rb[rb.invoice_line_id.isin(outcome(rb, res, gt)["fp"])]
    assert len(fp) > 0                                        # the spike rule does notice the step change ...
    assert set(fp.finding_type) == {"HISTORICAL_PRICE_SPIKE"}  # ... but nothing monetary fires
    assert (fp.potential_exposure == 0).all() and set(fp.severity) == {"LOW"}


# ------------------------------------------------------------------ 7. returns
def test_returns_flags_are_confined_to_returned_quantities(base, gt):
    d = copy.deepcopy(base)
    g = d.goods_receipts
    pick = g.sample(frac=0.05, random_state=3)
    ret = pick.assign(receipt_id=pick.receipt_id + "-RET",
                      quantity_received=-(pick.quantity_received * 0.2).round().clip(lower=1),
                      receipt_date=pick.receipt_date + pd.Timedelta(days=10))
    d.goods_receipts = pd.concat([g, ret], ignore_index=True)
    returned = ret.groupby("po_line_id").quantity_received.sum().abs()
    res, rb = detect(d)
    o = outcome(rb, res, gt)
    assert not o["missed"]
    fpq = rb[rb.invoice_line_id.isin(o["fp"])]
    assert set(fpq.finding_type) == {"QUANTITY_DISCREPANCY"}
    assert set(fpq.po_line_id) <= set(returned.index)
    flagged_units = fpq.details.map(lambda s: __import__("json").loads(s)["excess_quantity"]).groupby(fpq.po_line_id).sum()
    assert (flagged_units <= returned.reindex(flagged_units.index)).all()
    # loader surfaces the negative receipts as a data-quality issue for review
    assert any("non-positive receipt" in i["check"] for i in res.data.quality_issues)


# ------------------------------------------------------------------ metamorphic invariances
def _key(rb):
    return rb[["finding_id", "potential_exposure", "paid_exposure", "confirmed_loss"]].sort_values("finding_id").reset_index(drop=True)


def test_row_order_does_not_change_findings(base, baseline):
    d = copy.deepcopy(base)
    for t in ["invoices", "goods_receipts", "purchase_orders", "po_amendments"]:
        setattr(d, t, getattr(d, t).sample(frac=1.0, random_state=11).reset_index(drop=True))
    _, rb = detect(d)
    pd.testing.assert_frame_equal(_key(rb), _key(baseline.findings[baseline.findings.detection_method == "RULE_BASED"]))


def test_later_transactions_do_not_change_earlier_history_findings(base, baseline):
    d = copy.deepcopy(base)
    later = d.invoices.sample(200, random_state=5).copy()
    later["invoice_id"] = "LATE-" + later.invoice_id
    later["invoice_line_id"] = "LATE-" + later.invoice_line_id
    later["invoice_number"] = "LATE-" + later.invoice_number
    later["invoice_date"] = d.invoices.invoice_date.max() + pd.Timedelta(days=400)
    later["unit_price"] = later.unit_price * 3
    later["line_amount"] = (later.quantity_invoiced * later.unit_price).round(2)
    d.invoices = pd.concat([d.invoices, later], ignore_index=True)
    _, rb = detect(d)
    old = baseline.findings[(baseline.findings.detection_method == "RULE_BASED")
                            & (baseline.findings.finding_type == "HISTORICAL_PRICE_SPIKE")]
    new = rb[(rb.finding_type == "HISTORICAL_PRICE_SPIKE") & ~rb.invoice_line_id.str.startswith("LATE-")]
    pd.testing.assert_frame_equal(_key(new), _key(old))
