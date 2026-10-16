"""Load and check the procurement tables used by the detection engine.

The production pipeline reads ONLY the five procurement tables. The
ground-truth file is evaluation data and is loaded exclusively by
``leak_hunter.evaluation``; this module refuses to load it and rejects any
procurement table that contains label-like columns.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

GROUND_TRUTH_FILE = "ground_truth.csv"

REQUIRED_COLUMNS: dict[str, list[str]] = {
    "suppliers": ["supplier_id", "supplier_name", "category", "country"],
    "purchase_orders": ["po_id", "po_line_id", "supplier_id", "po_date", "item_id", "item_description",
                        "quantity_ordered", "unit_price", "currency", "approved_amendment_id"],
    "po_amendments": ["amendment_id", "po_id", "po_line_id", "amendment_date", "amendment_type", "status",
                      "previous_unit_price", "new_unit_price", "previous_quantity", "new_quantity"],
    "goods_receipts": ["receipt_id", "po_id", "po_line_id", "receipt_date", "item_id", "quantity_received"],
    "invoices": ["invoice_id", "invoice_number", "invoice_line_id", "supplier_id", "po_id", "po_line_id",
                 "invoice_date", "item_id", "quantity_invoiced", "unit_price", "currency", "payment_status",
                 "amount_paid"],
}
OPTIONAL_TABLES = {"po_amendments"}
PRIMARY_KEYS = {"suppliers": "supplier_id", "purchase_orders": "po_line_id", "po_amendments": "amendment_id",
                "goods_receipts": "receipt_id", "invoices": "invoice_line_id"}
DATE_COLUMNS = {"purchase_orders": ["po_date"], "po_amendments": ["amendment_date"],
                "goods_receipts": ["receipt_date"], "invoices": ["invoice_date"]}
NUMERIC_COLUMNS = {"purchase_orders": ["quantity_ordered", "unit_price"],
                   "po_amendments": ["previous_unit_price", "new_unit_price", "previous_quantity", "new_quantity"],
                   "goods_receipts": ["quantity_received"],
                   "invoices": ["quantity_invoiced", "unit_price", "amount_paid"]}
ID_COLUMNS = ["supplier_id", "po_id", "po_line_id", "item_id", "invoice_id", "invoice_line_id", "invoice_number",
              "receipt_id", "amendment_id", "approved_amendment_id"]

# Columns that must never appear in procurement inputs (ground-truth / label leakage).
LABEL_COLUMNS = {"is_anomaly", "primary_anomaly_type", "all_applicable_anomaly_types", "expected_exposure",
                 "exposure_currency", "expected_evidence_ids", "case_description", "exposure_basis",
                 "exposure_paid_to_date", "duplicate_group_id", "hist_n_prior", "hist_baseline_median",
                 "hist_price_ratio", "scenario_tags"}


class DataLoadError(Exception):
    """Raised when an input file or column is missing or unusable."""


@dataclass
class ProcurementData:
    suppliers: pd.DataFrame
    purchase_orders: pd.DataFrame
    po_amendments: pd.DataFrame
    goods_receipts: pd.DataFrame
    invoices: pd.DataFrame
    data_dir: Path | None = None
    quality_issues: list[dict] = field(default_factory=list)

    def tables(self) -> dict[str, pd.DataFrame]:
        return {k: getattr(self, k) for k in REQUIRED_COLUMNS}


def default_data_dir() -> Path:
    env = os.environ.get("LEAK_HUNTER_DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent / "Dataset" / "Data"


def _empty_amendments() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in REQUIRED_COLUMNS["po_amendments"]})


def prepare_table(name: str, df: pd.DataFrame, source: str = "") -> pd.DataFrame:
    """Validate columns and coerce types for one table (also used by tests)."""
    where = f" in {source}" if source else f" in table '{name}'"
    leaked = sorted(LABEL_COLUMNS.intersection(df.columns))
    if leaked:
        raise DataLoadError(f"Label/ground-truth columns {leaked} found{where}. Procurement inputs must not "
                            "contain evaluation labels.")
    missing = [c for c in REQUIRED_COLUMNS[name] if c not in df.columns]
    if missing:
        raise DataLoadError(f"Missing required column(s) {missing}{where}. Found: {list(df.columns)}")
    df = df.copy()
    for c in ID_COLUMNS:
        if c in df.columns:
            df[c] = df[c].astype("object").where(df[c].notna(), None)
            df[c] = df[c].map(lambda v: None if v is None or (isinstance(v, str) and v.strip() == "") else str(v).strip())
    for c in DATE_COLUMNS.get(name, []):
        parsed = pd.to_datetime(df[c], errors="coerce")
        bad = parsed.isna() & df[c].notna()
        if bad.any():
            raise DataLoadError(f"Column '{c}'{where} has {int(bad.sum())} unparseable date(s), "
                                f"e.g. {df.loc[bad, c].head(3).tolist()}")
        df[c] = parsed
    for c in NUMERIC_COLUMNS.get(name, []):
        num = pd.to_numeric(df[c], errors="coerce")
        bad = num.isna() & df[c].notna() & (df[c].astype(str).str.strip() != "")
        if bad.any():
            raise DataLoadError(f"Column '{c}'{where} has {int(bad.sum())} non-numeric value(s), "
                                f"e.g. {df.loc[bad, c].head(3).tolist()}")
        df[c] = num.astype(float)
    pk = PRIMARY_KEYS[name]
    if df[pk].isna().any():
        raise DataLoadError(f"Primary key '{pk}'{where} has missing values.")
    if df[pk].duplicated().any():
        raise DataLoadError(f"Primary key '{pk}'{where} is not unique, e.g. "
                            f"{df.loc[df[pk].duplicated(), pk].head(3).tolist()}")
    return df


def check_relationships(d: ProcurementData) -> list[dict]:
    """Referential and value checks. Issues are reported, not raised: rules skip
    what they cannot evaluate and the run summary lists every issue."""
    issues: list[dict] = []

    def add(check: str, mask: pd.Series, ids: pd.Series):
        n = int(mask.sum())
        if n:
            issues.append({"check": check, "count": n, "examples": ids[mask].head(5).tolist()})

    inv, po, gr, am, sup = d.invoices, d.purchase_orders, d.goods_receipts, d.po_amendments, d.suppliers
    pox = po.set_index("po_line_id")
    add("invoice supplier_id not in suppliers", ~inv.supplier_id.isin(sup.supplier_id), inv.invoice_line_id)
    add("invoice po_line_id not in purchase_orders", ~inv.po_line_id.isin(po.po_line_id), inv.invoice_line_id)
    known = inv.po_line_id.isin(po.po_line_id)
    add("invoice supplier differs from PO supplier",
        known & (inv.po_line_id.map(pox.supplier_id) != inv.supplier_id), inv.invoice_line_id)
    add("invoice item differs from PO line item", known & (inv.po_line_id.map(pox.item_id) != inv.item_id),
        inv.invoice_line_id)
    add("receipt po_line_id not in purchase_orders", ~gr.po_line_id.isin(po.po_line_id), gr.receipt_id)
    add("receipt item differs from PO line item",
        gr.po_line_id.isin(po.po_line_id) & (gr.po_line_id.map(pox.item_id) != gr.item_id), gr.receipt_id)
    add("amendment po_line_id not in purchase_orders", ~am.po_line_id.isin(po.po_line_id), am.amendment_id)
    ref = po.approved_amendment_id.notna()
    st = po.approved_amendment_id.map(am.set_index("amendment_id").status) if len(am) else pd.Series(index=po.index)
    add("PO approved_amendment_id not found or not Approved",
        ref & (st.astype(str).str.casefold() != "approved"), po.po_line_id)
    add("non-positive invoice quantity", inv.quantity_invoiced <= 0, inv.invoice_line_id)
    add("non-positive invoice unit price", inv.unit_price <= 0, inv.invoice_line_id)
    add("non-positive receipt quantity", gr.quantity_received <= 0, gr.receipt_id)
    add("non-positive PO unit price", po.unit_price <= 0, po.po_line_id)
    add("receipt dated before PO date",
        gr.po_line_id.isin(po.po_line_id) & (gr.receipt_date < gr.po_line_id.map(pox.po_date)), gr.receipt_id)
    add("invoice dated before PO date", known & (inv.invoice_date < inv.po_line_id.map(pox.po_date)),
        inv.invoice_line_id)
    cur = {c for c in pd.concat([inv.currency, po.currency]).dropna().unique()}
    if len(cur) > 1:
        issues.append({"check": "more than one currency (no FX conversion is applied)", "count": len(cur),
                       "examples": sorted(cur)})
    return issues


def load_procurement_data(data_dir: str | Path | None = None) -> ProcurementData:
    """Load the five procurement tables (never the ground truth)."""
    data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
    if not data_dir.is_dir():
        raise DataLoadError(f"Data directory not found: {data_dir}. Pass --data-dir or set LEAK_HUNTER_DATA_DIR.")
    frames = {}
    for name in REQUIRED_COLUMNS:
        path = data_dir / f"{name}.csv"
        if path.name == GROUND_TRUTH_FILE:  # defensive: never read labels here
            raise DataLoadError("The detection pipeline must not read ground_truth.csv")
        if not path.is_file():
            if name in OPTIONAL_TABLES:
                frames[name] = _empty_amendments()
                continue
            raise DataLoadError(f"Required input file not found: {path}")
        raw = pd.read_csv(path, dtype={c: "object" for c in ID_COLUMNS}, keep_default_na=True)
        frames[name] = prepare_table(name, raw, source=str(path))
    d = ProcurementData(**frames, data_dir=data_dir)
    if "po_amendments" in OPTIONAL_TABLES and not (data_dir / "po_amendments.csv").is_file():
        d.quality_issues.append({"check": "po_amendments.csv missing: no amendments applied", "count": 1,
                                 "examples": []})
    d.quality_issues.extend(check_relationships(d))
    return d


def from_frames(**frames: pd.DataFrame) -> ProcurementData:
    """Build ProcurementData from in-memory frames (tests, notebooks)."""
    out = {}
    for name in REQUIRED_COLUMNS:
        df = frames.get(name)
        if df is None:
            if name in OPTIONAL_TABLES:
                df = _empty_amendments()
            else:
                raise DataLoadError(f"Missing table '{name}'")
        out[name] = prepare_table(name, df)
    d = ProcurementData(**out)
    d.quality_issues.extend(check_relationships(d))
    return d
