"""Shared fixtures: a tiny procurement-data builder and the real repository dataset."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leak_hunter.config import DetectionConfig  # noqa: E402
from leak_hunter.data_loader import default_data_dir, from_frames  # noqa: E402
from leak_hunter.history import price_history  # noqa: E402
from leak_hunter.reconciliation import build_line_view  # noqa: E402


from dataclasses import replace as _replace  # noqa: E402

LATE_AUDIT = _replace(DetectionConfig(), rules=_replace(DetectionConfig().rules, as_of_date="2027-06-30"))


class Builder:
    """Assemble small, explicit scenarios. Dates are 'YYYY-MM-DD' strings."""

    def __init__(self):
        self.pos, self.rcpts, self.invs, self.amds = [], [], [], []
        self.sups = {}

    def po(self, po_line_id, supplier="SUP-1", item="ITM-1", qty=100, price=10.0, date="2026-01-01",
           approved_amendment_id=None):
        self.sups[supplier] = True
        self.pos.append(dict(po_id=po_line_id.rsplit("-", 1)[0], po_line_id=po_line_id, supplier_id=supplier,
                             po_date=date, item_id=item, item_description=f"desc {item}", quantity_ordered=qty,
                             unit_price=price, currency="USD", approved_amendment_id=approved_amendment_id))
        return self

    def receipt(self, receipt_id, po_line_id, qty, date):
        po = next(p for p in self.pos if p["po_line_id"] == po_line_id)
        self.rcpts.append(dict(receipt_id=receipt_id, po_id=po["po_id"], po_line_id=po_line_id, receipt_date=date,
                               item_id=po["item_id"], quantity_received=qty))
        return self

    def invoice(self, line_id, po_line_id, qty, price, date, number=None, status="Unpaid", paid=0.0):
        po = next(p for p in self.pos if p["po_line_id"] == po_line_id)
        inv_id = line_id.rsplit("-", 1)[0]
        self.invs.append(dict(invoice_id=inv_id, invoice_number=number or f"N-{inv_id}", invoice_line_id=line_id,
                              supplier_id=po["supplier_id"], po_id=po["po_id"], po_line_id=po_line_id,
                              invoice_date=date, item_id=po["item_id"], quantity_invoiced=qty, unit_price=price,
                              currency="USD", line_amount=round(qty * price, 2), payment_status=status,
                              amount_paid=paid))
        return self

    def amendment(self, amendment_id, po_line_id, date, kind="PRICE", status="Approved", new_price=None,
                  new_qty=None):
        po = next(p for p in self.pos if p["po_line_id"] == po_line_id)
        self.amds.append(dict(amendment_id=amendment_id, po_id=po["po_id"], po_line_id=po_line_id,
                              amendment_date=date, amendment_type=kind, status=status,
                              previous_unit_price=po["unit_price"] if kind == "PRICE" else None,
                              new_unit_price=new_price, previous_quantity=po["quantity_ordered"] if kind != "PRICE" else None,
                              new_quantity=new_qty, decided_by="Category Manager"))
        if status == "Approved":
            po["approved_amendment_id"] = amendment_id
        return self

    def data(self):
        sup = pd.DataFrame([dict(supplier_id=s, supplier_name=f"Name {s}", category="Cat", country="X",
                                 supplier_tier="Small") for s in self.sups])
        am = pd.DataFrame(self.amds) if self.amds else None
        return from_frames(suppliers=sup, purchase_orders=pd.DataFrame(self.pos),
                           goods_receipts=pd.DataFrame(self.rcpts), invoices=pd.DataFrame(self.invs),
                           po_amendments=am)

    def view(self, config=None):
        # Default: audit long after the scenario, so the posting-lag grace period never applies.
        # Grace-period tests pass their own config with an explicit as_of_date.
        cfg = config or LATE_AUDIT
        v = build_line_view(self.data(), cfg)
        return v, price_history(v, cfg)


@pytest.fixture
def builder():
    return Builder()


REAL_DATA_DIR = default_data_dir()


@pytest.fixture(scope="session")
def real_data_dir():
    if not (REAL_DATA_DIR / "invoices.csv").is_file():
        pytest.skip(f"Repository dataset not found at {REAL_DATA_DIR}")
    return REAL_DATA_DIR


@pytest.fixture(scope="session")
def real_result(real_data_dir):
    from leak_hunter.pipeline import run_detection
    return run_detection(data_dir=real_data_dir)


@pytest.fixture(scope="session")
def ground_truth(real_data_dir):
    from leak_hunter.evaluation import load_ground_truth
    return load_ground_truth(real_data_dir)
