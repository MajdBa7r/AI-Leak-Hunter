"""Money rounding, payment proration, duplicate-payment logic and exposure de-duplication."""
from __future__ import annotations

import numpy as np
import pandas as pd

from leak_hunter.exposure import (DUP, HIST, PRICE, QTY, from_cents, incident_table, line_exposure, prorate_paid,
                                  realized_duplicate_payments, to_cents)


def test_cents_rounding_is_half_up_and_exact():
    assert to_cents(1.005) == 101
    assert to_cents(0.125) == 13
    assert list(to_cents([10.0, 2.675])) == [1000, 268]
    assert from_cents(101) == 1.01


def test_prorate_paid():
    assert prorate_paid(1000, 0, 5000) == 0
    assert prorate_paid(1000, 5000, 5000) == 1000
    assert prorate_paid(1000, 2500, 5000) == 500
    assert prorate_paid(1000, 100, 0) == 0          # no division by zero


def test_realized_duplicate_payments():
    assert realized_duplicate_payments(1000, [1000, 1000], [1000]) == [1000]   # paid twice
    assert realized_duplicate_payments(1000, [1000, 0], [1000]) == [0]        # repeat unpaid
    assert realized_duplicate_payments(1000, [600, 600], [1000]) == [200]     # partial + partial
    assert realized_duplicate_payments(1000, [0, 1000], [1000]) == [0]        # only one payment so far
    assert realized_duplicate_payments(1000, [1000, 1000, 1000], [1000, 1000]) == [1000, 1000]  # triplicate


def F(line, ftype, pot, paid=0.0, conf=0.0, inc=None, method="RULE_BASED"):
    return dict(finding_id=f"F-{ftype[:3]}-{line}", incident_id=inc or f"INC-{line}", finding_type=ftype,
                detection_method=method, invoice_id=line[:-3], invoice_line_id=line, supplier_id="SUP-1",
                severity="LOW", potential_exposure=pot, paid_exposure=paid, confirmed_loss=conf,
                anomaly_score=np.nan, currency="USD")


def test_line_exposure_deduplicates_overlapping_rules():
    f = pd.DataFrame([
        F("L-1-01", PRICE, 50.0), F("L-1-01", HIST, 50.0),                        # same contractual excess
        F("L-2-01", QTY, 30.0), F("L-2-01", PRICE, 20.0),                         # additive components
        F("L-3-01", DUP, 200.0, 200.0, 200.0, inc="INC-DUP-X"), F("L-3-01", QTY, 40.0, inc="INC-DUP-X"),
        F("L-4-01", DUP, 0.0, inc="INC-DUP-X"),                                  # original entry
        F("L-5-01", HIST, 0.0),                                                   # unusual, no basis
        F("L-6-01", "UNUSUAL_TRANSACTION", 0.0, method="ISOLATION_FOREST"),
    ])
    le = line_exposure(f).set_index("invoice_line_id").line_exposure.to_dict()
    assert le == {"L-1-01": 50.0, "L-2-01": 50.0, "L-3-01": 200.0, "L-4-01": 0.0, "L-5-01": 0.0}
    inc = incident_table(f).set_index("incident_id")
    assert inc.loc["INC-DUP-X", "potential_exposure"] == 200.0
    assert inc.loc["INC-DUP-X", "confirmed_loss"] == 200.0
    assert inc.loc["INC-L-6-01", "potential_exposure"] == 0.0
    assert inc.potential_exposure.sum() == 300.0                 # vs naive 390
    assert round(f.potential_exposure.sum(), 2) == 390.0


def test_rule_exposures_match_documented_formulas(builder):
    from leak_hunter.detection_rules import rule_po_price_mismatch, rule_quantity_discrepancy
    builder.po("PO-1-01", qty=100, price=12.34).receipt("GR-1", "PO-1-01", 90, "2026-01-05")
    builder.invoice("INV-1-01", "PO-1-01", 97, 12.99, "2026-01-06", status="Paid", paid=round(97 * 12.99, 2))
    v, _ = builder.view()
    q = rule_quantity_discrepancy(v).iloc[0]
    p = rule_po_price_mismatch(v).iloc[0]
    assert q.potential_exposure == round(7 * 12.34, 2)            # excess x approved price
    assert p.potential_exposure == round((12.99 - 12.34) * 97, 2)  # price diff x quantity invoiced
