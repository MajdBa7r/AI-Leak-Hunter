"""Unit tests for the four deterministic rules on small, explicit scenarios."""
from __future__ import annotations

from dataclasses import replace

import pytest

from leak_hunter.config import DetectionConfig
from leak_hunter.detection_rules import (rule_duplicate_invoices, rule_historical_price_spike,
                                         rule_po_price_mismatch, rule_quantity_discrepancy)
from leak_hunter.exposure import incident_table
from leak_hunter.findings import build_findings, link_related, validate_findings
from leak_hunter.reconciliation import fuzzy_invoice_key, normalize_invoice_number, osa_distance


def Builder_fresh():
    from conftest import Builder
    b = Builder()
    b.pipe = lambda fn, **k: fn(b, **k)
    return b


def by_line(hits):
    return {r.invoice_line_id: r for r in hits.itertuples()}


# ------------------------------------------------------------------ Rule A: duplicate invoices
def dup_case(b, status1="Paid", paid1=1000.0, status2="Unpaid", paid2=0.0, number2="inv 2026/0001"):
    b.po("PO-1-01", qty=100, price=10.0).receipt("GR-1", "PO-1-01", 100, "2026-01-05")
    b.invoice("INV-1-01", "PO-1-01", 100, 10.0, "2026-01-06", number="INV-2026-0001", status=status1, paid=paid1)
    b.invoice("INV-2-01", "PO-1-01", 100, 10.0, "2026-01-20", number=number2, status=status2, paid=paid2)
    return b


def test_normalize_invoice_number():
    assert normalize_invoice_number(" inv-2026/0001 ") == "INV20260001"
    assert normalize_invoice_number("INV 2026.0001") == "INV20260001"
    assert normalize_invoice_number(None) == ""
    assert normalize_invoice_number("INV-0012", strip_leading_zeros=True) == "INV12"


def test_leading_zero_stripping_works_inside_separated_numbers():
    # v1 bug: separators were removed first, so "26" and "012345" merged and the zero survived
    assert normalize_invoice_number("SI/26/012345", strip_leading_zeros=True) == "SI2612345"
    assert normalize_invoice_number("SI/26/12345", strip_leading_zeros=True) == "SI2612345"
    assert normalize_invoice_number("202604-000000", strip_leading_zeros=True) == "2026040"


def test_fuzzy_key_and_edit_distance():
    assert fuzzy_invoice_key("INV-2O26-OO12") == fuzzy_invoice_key("inv 2026 0012") == "1NV202612"
    assert osa_distance("ABC123", "ABC123") == 0
    assert osa_distance("ABC123", "ABC132") == 1      # adjacent swap
    assert osa_distance("ABC123", "ABC123A") == 1     # insertion
    assert osa_distance("ABC123", "ABD124") == 2


@pytest.mark.parametrize("repeat_number", ["INV-2026-0001-A",   # suffix (insertion)
                                           "INV-2026-0010",     # adjacent swap
                                           "INV-2O26-0001",     # letter O for zero
                                           "INV-2026-001",      # dropped leading zero
                                           "1NV-2026-0001",     # digit 1 for letter I
                                           "INV-2026-00001",    # extra leading zero
                                           "INV 2026 0001"])    # separators
def test_fuzzy_duplicate_variants_are_matched(builder, repeat_number):
    v, _ = dup_case(builder, number2=repeat_number).view()
    hits = by_line(rule_duplicate_invoices(v))
    assert set(hits) == {"INV-1-01", "INV-2-01"}
    assert hits["INV-2-01"].details["match_type"] in {"FUZZY", "EXACT"}
    assert hits["INV-2-01"].potential_exposure == 1000.0


def test_fuzzy_match_requires_identical_amount_and_date_window(builder):
    builder.po("PO-1-01", qty=200, price=10.0).receipt("GR-1", "PO-1-01", 200, "2026-01-05")
    builder.invoice("INV-1-01", "PO-1-01", 100, 10.0, "2026-01-06", number="INV-2026-0001")
    builder.invoice("INV-2-01", "PO-1-01", 99, 10.0, "2026-01-10", number="INV-2026-0001-B")  # one edit, other amount
    builder.po("PO-2-01", qty=100, price=10.0).receipt("GR-2", "PO-2-01", 100, "2026-01-05")
    builder.invoice("INV-3-01", "PO-2-01", 100, 10.0, "2026-06-01", number="INV-2026-0003")  # same amount, 146 days later
    v, _ = builder.view()
    assert rule_duplicate_invoices(v).empty


def test_recurring_identical_amount_sequential_invoices_are_not_duplicates(builder):
    # fixed monthly fee under one blanket PO line: same amount, consecutive numbers
    builder.po("PO-1-01", qty=6, price=500.0, date="2026-01-01")
    for m in range(1, 7):
        builder.receipt(f"GR-{m}", "PO-1-01", 1, f"2026-0{m}-01")
        builder.invoice(f"INV-{m}-01", "PO-1-01", 1, 500.0, f"2026-0{m}-02", number=f"FEE-2026-{1040 + m}")
    v, _ = builder.view()
    assert rule_duplicate_invoices(v).empty
    cfg = replace(DetectionConfig(), rules=replace(DetectionConfig().rules, dup_fuzzy_allow_digit_substitution=True,
                                                   as_of_date="2027-06-30"))
    v2, _ = builder.view(cfg)
    assert len(rule_duplicate_invoices(v2, cfg)) > 0     # the reason the guard exists


def test_fuzzy_matching_can_be_disabled(builder):
    cfg = replace(DetectionConfig(), rules=replace(DetectionConfig().rules, dup_fuzzy=False, as_of_date="2027-06-30"))
    v, _ = dup_case(builder, number2="INV-2026-0001-A").view(cfg)
    assert rule_duplicate_invoices(v, cfg).empty


def test_duplicate_detects_normalised_numbers_and_links_both_entries(builder):
    v, _ = dup_case(builder).view()
    hits = by_line(rule_duplicate_invoices(v))
    assert set(hits) == {"INV-1-01", "INV-2-01"}
    orig, rep = hits["INV-1-01"], hits["INV-2-01"]
    assert orig.details["role"] == "ORIGINAL" and rep.details["role"] == "REPEAT"
    assert orig.potential_exposure == 0 and rep.potential_exposure == 1000.0
    assert set(rep.evidence["invoices"]) == {"INV-1", "INV-2"}
    assert rep.evidence["suppliers"] == ["SUP-1"]
    assert rep.details["same_quantity_and_price_as_original"] is True


def test_same_number_different_supplier_is_not_a_duplicate(builder):
    builder.po("PO-1-01", supplier="SUP-1").po("PO-2-01", supplier="SUP-2")
    builder.receipt("GR-1", "PO-1-01", 100, "2026-01-05").receipt("GR-2", "PO-2-01", 100, "2026-01-05")
    builder.invoice("INV-1-01", "PO-1-01", 100, 10.0, "2026-01-06", number="A-1")
    builder.invoice("INV-2-01", "PO-2-01", 100, 10.0, "2026-01-07", number="A-1")
    v, _ = builder.view()
    assert rule_duplicate_invoices(v).empty


@pytest.mark.parametrize("s1,p1,s2,p2,confirmed,assessment", [
    ("Paid", 1000.0, "Paid", 1000.0, 1000.0, "CONFIRMED_DUPLICATE_PAYMENT"),
    ("Paid", 1000.0, "Unpaid", 0.0, 0.0, "REPEAT_NOT_YET_PAID"),
    ("Partially Paid", 600.0, "Partially Paid", 600.0, 200.0, "CONFIRMED_DUPLICATE_PAYMENT"),
    ("Unpaid", 0.0, "Paid", 1000.0, 0.0, "REPEAT_PAID_ORIGINAL_NOT_FULLY_PAID"),
])
def test_duplicate_invoice_vs_confirmed_duplicate_payment(builder, s1, p1, s2, p2, confirmed, assessment):
    v, _ = dup_case(builder, s1, p1, s2, p2).view()
    rep = by_line(rule_duplicate_invoices(v))["INV-2-01"]
    assert rep.potential_exposure == 1000.0          # duplicate invoice: potential exposure always
    assert rep.confirmed_loss == confirmed           # duplicate payment: only when payments prove it
    assert rep.details["payment_assessment"] == assessment


def test_duplicate_exposure_not_double_counted(builder):
    v, h = dup_case(builder, "Paid", 1000.0, "Paid", 1000.0).view()
    hits = rule_duplicate_invoices(v)
    # the repeat also over-bills the received quantity -> a second finding on the same line
    qty = rule_quantity_discrepancy(v)
    assert set(qty.invoice_line_id) == {"INV-2-01"}
    import pandas as pd
    f = link_related(build_findings(pd.concat([hits, qty], ignore_index=True), v))
    validate_findings(f)
    inc = incident_table(f)
    assert len(inc) == 1                              # one pair = one incident
    assert inc.potential_exposure.iloc[0] == 1000.0   # not 1000 + 1000 (qty)
    assert inc.confirmed_loss.iloc[0] == 1000.0
    assert round(f.potential_exposure.sum(), 2) == 2000.0  # the naive sum would double count


# ------------------------------------------------------------------ Rule B: quantity discrepancies
def test_multiple_invoices_share_po_line_without_false_positive(builder):
    builder.po("PO-1-01", qty=100).receipt("GR-1", "PO-1-01", 60, "2026-01-05").receipt("GR-2", "PO-1-01", 40, "2026-01-20")
    builder.invoice("INV-1-01", "PO-1-01", 60, 10.0, "2026-01-06")
    builder.invoice("INV-2-01", "PO-1-01", 40, 10.0, "2026-01-21")
    v, _ = builder.view()
    assert rule_quantity_discrepancy(v).empty
    # each invoice individually is <= the receipt total, AND the cumulative total is too
    assert v.set_index("invoice_line_id").cum_invoiced.to_dict() == {"INV-1-01": 60, "INV-2-01": 100}


def test_cumulative_over_billing_across_invoices_is_flagged_on_the_excess_line(builder):
    # each invoice alone (60, 60) is within the 100 received, together they exceed it
    builder.po("PO-1-01", qty=120).receipt("GR-1", "PO-1-01", 50, "2026-01-05").receipt("GR-2", "PO-1-01", 50, "2026-01-25")
    builder.invoice("INV-1-01", "PO-1-01", 60, 10.0, "2026-01-06")
    builder.invoice("INV-2-01", "PO-1-01", 60, 10.0, "2026-01-26", status="Partially Paid", paid=300.0)
    v, _ = builder.view()
    hits = by_line(rule_quantity_discrepancy(v))
    assert set(hits) == {"INV-2-01"}
    h = hits["INV-2-01"]
    assert h.details["excess_quantity"] == 20 and h.potential_exposure == 200.0
    assert h.paid_exposure == 100.0                    # 300/600 of the line paid -> half the exposure
    assert h.evidence["receipts"] == ["GR-1", "GR-2"]
    assert set(h.evidence["invoice_lines"]) == {"INV-1-01", "INV-2-01"}


def test_partial_delivery_completed_later_is_not_flagged(builder):
    builder.po("PO-1-01", qty=100).receipt("GR-1", "PO-1-01", 30, "2026-01-05")
    builder.receipt("GR-2", "PO-1-01", 30, "2026-01-20").receipt("GR-3", "PO-1-01", 40, "2026-02-10")
    builder.invoice("INV-1-01", "PO-1-01", 100, 10.0, "2026-01-06")   # billed in full at first delivery
    v, _ = builder.view()
    assert rule_quantity_discrepancy(v).empty
    assert v.received_as_of_invoice.iloc[0] == 30 and v.received_total.iloc[0] == 100


def test_later_receipt_does_not_hide_real_excess(builder):
    builder.po("PO-1-01", qty=100).receipt("GR-1", "PO-1-01", 50, "2026-01-05").receipt("GR-2", "PO-1-01", 40, "2026-02-01")
    builder.invoice("INV-1-01", "PO-1-01", 100, 10.0, "2026-01-06")
    v, _ = builder.view()
    h = by_line(rule_quantity_discrepancy(v))["INV-1-01"]
    assert h.details["excess_quantity"] == 10
    assert h.details["receipts_dated_after_invoice"] == ["GR-2"]


@pytest.mark.parametrize("status,flagged", [("Approved", False), ("Rejected", True)])
def test_quantity_amendment_approved_vs_rejected(builder, status, flagged):
    builder.po("PO-1-01", qty=100).receipt("GR-1", "PO-1-01", 120, "2026-01-05")
    builder.amendment("AMD-1", "PO-1-01", "2026-01-03", kind="QUANTITY", status=status, new_qty=120)
    builder.invoice("INV-1-01", "PO-1-01", 120, 10.0, "2026-01-06")
    v, _ = builder.view()
    hits = rule_quantity_discrepancy(v)
    assert (not hits.empty) == flagged
    if flagged:
        assert hits.details.iloc[0]["limit"] == "EXCEEDS_APPROVED_QUANTITY"
        assert hits.potential_exposure.iloc[0] == 200.0


# ------------------------------------------------------------------ Rule B: grace period
def grace_cfg(as_of, grace=30):
    return replace(DetectionConfig(), rules=replace(DetectionConfig().rules, as_of_date=as_of, qty_grace_days=grace))


def grace_case(b):
    b.po("PO-1-01", qty=100).receipt("GR-1", "PO-1-01", 60, "2026-03-01")
    b.invoice("INV-1-01", "PO-1-01", 100, 10.0, "2026-03-02")      # 40 units not posted yet
    return b


def test_recent_billing_ahead_of_receipt_is_pending_not_flagged(builder):
    from leak_hunter.detection_rules import pending_quantity_checks
    cfg = grace_cfg("2026-03-20")
    v, _ = grace_case(builder).view(cfg)
    assert rule_quantity_discrepancy(v, cfg).empty
    p = pending_quantity_checks(v, cfg)
    assert p.pending_units.tolist() == [40] and p.days_until_flag.tolist() == [12]
    assert p.amount_at_stake_if_unreceived.tolist() == [400.0]


def test_after_grace_period_missing_receipts_are_flagged(builder):
    from leak_hunter.detection_rules import pending_quantity_checks
    cfg = grace_cfg("2026-04-15")
    v, _ = grace_case(builder).view(cfg)
    h = rule_quantity_discrepancy(v, cfg).iloc[0]
    assert h.details["excess_quantity"] == 40 and h.details["within_grace_period"] is False
    assert pending_quantity_checks(v, cfg).empty


def test_billing_beyond_approved_quantity_is_flagged_inside_grace(builder):
    from leak_hunter.detection_rules import pending_quantity_checks
    builder.po("PO-1-01", qty=100).receipt("GR-1", "PO-1-01", 60, "2026-03-01")
    builder.invoice("INV-1-01", "PO-1-01", 130, 10.0, "2026-03-02")
    cfg = grace_cfg("2026-03-10")
    v, _ = builder.view(cfg)
    h = rule_quantity_discrepancy(v, cfg).iloc[0]
    assert h.details["excess_quantity"] == 30 and h.potential_exposure == 300.0   # beyond the order: flag now
    assert pending_quantity_checks(v, cfg).pending_units.tolist() == [40]          # the rest may still arrive


# ------------------------------------------------------------------ Rule C: PO price mismatch
def price_case(b, amd_date="2026-01-03", status="Approved", new_price=11.0, inv_price=11.0):
    b.po("PO-1-01", qty=100, price=10.0).receipt("GR-1", "PO-1-01", 100, "2026-01-05")
    b.amendment("AMD-1", "PO-1-01", amd_date, kind="PRICE", status=status, new_price=new_price)
    b.invoice("INV-1-01", "PO-1-01", 100, inv_price, "2026-01-06", status="Paid", paid=100 * inv_price)
    return b


def test_approved_price_amendment_authorises_higher_price(builder):
    v, _ = price_case(builder).view()
    assert rule_po_price_mismatch(v).empty
    assert v.approved_unit_price.iloc[0] == 11.0 and v.applied_price_amendment_id.iloc[0] == "AMD-1"


def test_amendment_dated_after_invoice_is_not_applied(builder):
    v, _ = price_case(builder, amd_date="2026-01-10").view()
    h = rule_po_price_mismatch(v).iloc[0]
    assert h.details["approved_unit_price"] == 10.0 and h.potential_exposure == 100.0
    assert h.details["approved_amendments_dated_after_invoice"] == ["AMD-1"]


def test_rejected_price_amendment_is_not_authorised(builder):
    v, _ = price_case(builder, status="Rejected", new_price=12.0, inv_price=12.0).view()
    h = rule_po_price_mismatch(v).iloc[0]
    assert h.details["approved_unit_price"] == 10.0
    assert h.details["billed_at_unapproved_amendment_price"] == ["AMD-1"]
    assert h.potential_exposure == 200.0
    assert h.paid_exposure == 200.0 and h.confirmed_loss == 0.0   # paid, but not a confirmed loss
    assert "AMD-1" in h.evidence["amendments"]


def test_price_above_approved_amendment_is_flagged_against_amended_price(builder):
    v, _ = price_case(builder, new_price=11.0, inv_price=11.5).view()
    h = rule_po_price_mismatch(v).iloc[0]
    assert h.details["po_unit_price"] == 10.0 and h.details["approved_unit_price"] == 11.0
    assert h.potential_exposure == 50.0


@pytest.mark.parametrize("approved,invoiced,qty,flagged", [
    (10.00, 10.20, 100, False),    # 2.0% exactly: not ABOVE 2%
    (10.00, 10.21, 100, True),     # 2.1% and $21 line overcharge
    (10.00, 10.50, 5, False),      # 5% but only $2.50 on the line: immaterial
    (10.00, 11.10, 5, True),       # 11% and $5.50
    (1000.00, 1015.00, 1, False),  # $15 but 1.5%: within contractual variance
    (1000.00, 1030.00, 1, True),   # $30 and 3%
])
def test_price_materiality_dual_threshold(builder, approved, invoiced, qty, flagged):
    builder.po("PO-1-01", qty=qty, price=approved).receipt("GR-1", "PO-1-01", qty, "2026-01-05")
    builder.invoice("INV-1-01", "PO-1-01", qty, invoiced, "2026-01-06")
    v, _ = builder.view()
    hits = rule_po_price_mismatch(v)
    assert (len(hits) == 1) == flagged
    if flagged:   # exposure is the FULL difference, not just the part above the thresholds
        assert hits.potential_exposure.iloc[0] == round((invoiced - approved) * qty, 2)


def test_materiality_configurable(builder):
    builder.po("PO-1-01", price=10.0).receipt("GR-1", "PO-1-01", 100, "2026-01-05")
    builder.invoice("INV-1-01", "PO-1-01", 100, 10.01, "2026-01-06")
    v, _ = builder.view()
    assert rule_po_price_mismatch(v).empty
    cfg = replace(DetectionConfig(), rules=replace(DetectionConfig().rules, price_materiality_abs=0.0,
                                                   price_materiality_pct=0.0))
    assert len(rule_po_price_mismatch(v, cfg)) == 1


def test_amendment_just_after_invoice_is_borderline_low_confidence(builder):
    import json as _j
    v, _ = price_case(builder, amd_date="2026-01-08").view()        # Tue 6 Jan invoice, Thu 8 Jan approval
    h = rule_po_price_mismatch(v).iloc[0]
    assert h.details["confidence"] == "LOW" and h.details["ambiguity"] == "BORDERLINE_AMENDMENT_TIMING"
    v2, _ = Builder_fresh().pipe(price_case, amd_date="2026-01-20").view()   # two weeks later: clean violation
    assert rule_po_price_mismatch(v2).iloc[0].details["confidence"] == "HIGH"


def test_pending_amendment_never_authorises_but_lowers_confidence(builder):
    v, _ = price_case(builder, status="Pending", new_price=12.0, inv_price=12.0).view()
    h = rule_po_price_mismatch(v).iloc[0]
    assert h.details["approved_unit_price"] == 10.0
    assert h.details["confidence"] == "MEDIUM" and h.details["ambiguity"] == "PENDING_AMENDMENT"


def test_quantity_materiality_ignores_tiny_or_low_value_excess(builder):
    builder.po("PO-1-01", qty=1000, price=10.0).receipt("GR-1", "PO-1-01", 1000, "2026-01-05")
    builder.invoice("INV-1-01", "PO-1-01", 1010, 10.0, "2026-01-06")     # 1% excess: below 1.5%
    builder.po("PO-2-01", qty=10, price=2.0).receipt("GR-2", "PO-2-01", 10, "2026-01-05")
    builder.invoice("INV-2-01", "PO-2-01", 12, 2.0, "2026-01-06")        # 17% but $4: low value
    builder.po("PO-3-01", qty=100, price=10.0).receipt("GR-3", "PO-3-01", 100, "2026-01-05")
    builder.invoice("INV-3-01", "PO-3-01", 110, 10.0, "2026-01-06")      # 9%, $100: material
    v, _ = builder.view()
    assert set(rule_quantity_discrepancy(v).invoice_line_id) == {"INV-3-01"}


def test_recurring_30_day_cadence_is_not_a_fuzzy_duplicate(builder):
    builder.po("PO-1-01", qty=200, price=10.0).receipt("GR-1", "PO-1-01", 200, "2026-01-05")
    builder.invoice("INV-1-01", "PO-1-01", 100, 10.0, "2026-01-06", number="RET-2026-0001")
    builder.invoice("INV-2-01", "PO-1-01", 100, 10.0, "2026-02-05", number="RET-2026-0001-A")   # exactly 30 days
    v, _ = builder.view()
    assert rule_duplicate_invoices(v).empty


def test_exact_number_is_a_duplicate_even_30_days_later(builder):
    v, _ = dup_case(builder, number2="INV-2026-0001").view()
    assert len(rule_duplicate_invoices(v)) == 2      # exemption applies to fuzzy matches only


def test_fuzzy_distance_two_requires_same_items(builder):
    builder.po("PO-1-01", item="ITM-1", qty=100, price=10.0).receipt("GR-1", "PO-1-01", 100, "2026-01-05")
    builder.po("PO-2-01", item="ITM-2", qty=100, price=10.0).receipt("GR-2", "PO-2-01", 100, "2026-01-05")
    builder.invoice("INV-1-01", "PO-1-01", 100, 10.0, "2026-01-06", number="AB-77123")
    builder.invoice("INV-2-01", "PO-2-01", 100, 10.0, "2026-01-12", number="AB-77123-XY")   # 2 edits, other item
    v, _ = builder.view()
    assert rule_duplicate_invoices(v).empty
    builder2 = Builder_fresh()
    builder2.po("PO-1-01", item="ITM-1", qty=200, price=10.0).receipt("GR-1", "PO-1-01", 200, "2026-01-05")
    builder2.invoice("INV-1-01", "PO-1-01", 100, 10.0, "2026-01-06", number="AB-77123")
    builder2.invoice("INV-2-01", "PO-1-01", 100, 10.0, "2026-01-12", number="AB-77123-XY")  # 2 edits, same item
    v2, _ = builder2.view()
    assert len(rule_duplicate_invoices(v2)) == 2


# ------------------------------------------------------------------ Rule D: historical price spike
def history_case(b, prior_prices, current_price, current_po_price=None, extra=()):
    """prior purchases on separate PO lines, one per month, then the current one on 2026-08-01."""
    for i, p in enumerate(prior_prices, start=1):
        pl = f"PO-{i}-01"
        b.po(pl, qty=10, price=p, date=f"2026-0{i}-01" if i < 10 else "2026-07-01")
        b.receipt(f"GR-{i}", pl, 10, f"2026-0{i}-02")
        b.invoice(f"INV-{i}-01", pl, 10, p, f"2026-0{i}-03")
    b.po("PO-99-01", qty=10, price=current_po_price if current_po_price is not None else current_price, date="2026-07-25")
    b.receipt("GR-99", "PO-99-01", 10, "2026-07-30")
    b.invoice("INV-99-01", "PO-99-01", 10, current_price, "2026-08-01")
    for k, (date, price) in enumerate(extra, start=200):
        pl = f"PO-{k}-01"
        b.po(pl, qty=10, price=price, date=date)
        b.receipt(f"GR-{k}", pl, 10, date)
        b.invoice(f"INV-{k}-01", pl, 10, price, date)
    return b


def test_historical_spike_flagged_with_zero_exposure_without_contract_basis(builder):
    v, h = history_case(builder, [10, 10, 10, 10, 10], 14.0).view()
    hits = by_line(rule_historical_price_spike(v, h))
    assert set(hits) == {"INV-99-01"}
    d = hits["INV-99-01"]
    assert d.details["n_prior_observations"] == 5 and d.details["price_ratio"] == 1.4
    assert d.potential_exposure == 0.0 and d.details["has_contractual_basis"] is False
    assert set(d.evidence["invoice_lines"]) >= {f"INV-{i}-01" for i in range(1, 6)}


def test_historical_spike_contractual_excess_when_above_po_price(builder):
    v, h = history_case(builder, [10, 10, 10, 10, 10], 14.0, current_po_price=10.0).view()
    d = by_line(rule_historical_price_spike(v, h))["INV-99-01"]
    assert d.potential_exposure == 40.0 and d.details["has_contractual_basis"] is True
    assert len(rule_po_price_mismatch(v)) == 1   # the same excess is also a Rule C finding


def test_historical_spike_requires_five_prior_observations(builder):
    v, h = history_case(builder, [10, 10, 10, 10], 20.0).view()
    assert rule_historical_price_spike(v, h).empty


def test_historical_baseline_ignores_future_and_same_day_transactions(builder):
    extra = [("2026-08-01", 100.0), ("2026-09-01", 100.0), ("2026-10-01", 100.0)]
    v, h = history_case(builder, [10, 10, 10, 10, 10], 14.0, extra=extra).view()
    row = h.loc[v.invoice_line_id == "INV-99-01"].iloc[0]
    assert row.hist_n_prior == 5 and row.hist_baseline == 10.0
    assert all(i.startswith("INV-") and int(i.split("-")[1]) <= 5 for i in row.hist_prior_line_ids)


def test_variable_price_history_widens_threshold(builder):
    v, h = history_case(builder, [8, 12, 7, 13, 10], 14.0).view()
    assert rule_historical_price_spike(v, h).empty             # 1.4x is normal for this history
    cfg = DetectionConfig()
    cfg = replace(cfg, rules=replace(cfg.rules, hist_adaptive=False))
    from leak_hunter.history import price_history
    assert len(rule_historical_price_spike(v, price_history(v, cfg), cfg)) == 1


def test_duplicate_reentries_are_not_independent_history(builder):
    history_case(builder, [10, 10, 10, 10], 14.0)
    builder.invoice("INV-50-01", "PO-4-01", 10, 10.0, "2026-04-20", number="N-INV-4")  # re-entry of INV-4
    v, h = builder.view()
    assert rule_historical_price_spike(v, h).empty
    cfg = DetectionConfig()
    cfg = replace(cfg, rules=replace(cfg.rules, hist_exclude_duplicate_reentries=False))
    from leak_hunter.history import price_history
    assert len(rule_historical_price_spike(v, price_history(v, cfg), cfg)) == 1
