#!/usr/bin/env python3
"""
build_unified_dataset.py
========================
Integrates the AI Leak Hunter source files in ./data/ into ONE CSV at invoice-line grain:

    data/ai_leak_hunter_unified.csv      5,000 rows, one per invoice line
    data/unified_schema.csv              column dictionary (source, role, missing-value rule)
    data/unified_validation_report.md    every validation check + join summary

Run after generate_synthetic_data.py:   python build_unified_dataset.py
The source CSVs are read, never modified.

Join plan (all left joins from invoices.csv, each key verified unique on the right side):
    invoices.csv        base table, 1 row per invoice_line_id
    ground_truth.csv    1:1   on invoice_line_id (+ invoice_id must agree)
    suppliers.csv       N:1   on supplier_id
    purchase_orders.csv N:1   on po_line_id (po_id, supplier_id, item_id must agree)
    po_amendments.csv   N:1   on po_line_id (at most one amendment of any status per PO line)
    goods_receipts.csv  N:M   aggregated to one row per po_line_id BEFORE joining, so receipts
                              never multiply invoice rows; individual receipts are kept as
                              ';'-separated lists in date order (lossless).

Column roles (see unified_schema.csv):
    key / feature   procurement data the detector may use
    derived         computed only from procurement data available at the invoice date
    label           ground truth: EVALUATION ONLY, drop before detection (all LABEL_COLUMNS)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT = os.path.join(DATA, "ai_leak_hunter_unified.csv")
N_ROWS, N_NORMAL, N_ANOM = 5000, 4500, 500
TYPE_COUNTS = {"PO_PRICE_MISMATCH": 170, "QUANTITY_DISCREPANCY": 140,
               "HISTORICAL_PRICE_SPIKE": 110, "DUPLICATE_INVOICE": 80}
HIST_MIN_PRIOR, HIST_RATIO = 5, 1.30
SEP = ";"

MONEY = ["unit_price", "line_amount", "amount_paid", "po_unit_price", "amendment_previous_unit_price",
         "amendment_new_unit_price", "approved_unit_price", "expected_exposure",
         "exposure_paid_to_date", "hist_baseline_median"]
DATES = ["invoice_date", "po_date", "amendment_date", "first_receipt_date", "last_receipt_date"]

# column -> (source, role, description, when missing)
SCHEMA = {
    "invoice_id": ("invoices", "key", "Internal invoice document ID", "never"),
    "invoice_line_id": ("invoices", "key", "Invoice line ID (primary key of this file)", "never"),
    "invoice_number": ("invoices", "feature", "Supplier's own invoice number (string)", "never"),
    "supplier_id": ("invoices", "key", "Supplier ID; equals the PO's supplier", "never"),
    "po_id": ("invoices", "key", "Purchase order ID", "never"),
    "po_line_id": ("invoices", "key", "Purchase order line ID", "never"),
    "item_id": ("invoices", "key", "Item ID; equals the PO line and receipt item", "never"),
    "invoice_date": ("invoices", "feature", "Invoice date", "never"),
    "quantity_invoiced": ("invoices", "feature", "Invoiced quantity", "never"),
    "unit_price": ("invoices", "feature", "Invoiced unit price, USD", "never"),
    "currency": ("invoices", "feature", "Invoice currency (USD)", "never"),
    "line_amount": ("invoices", "feature", "quantity_invoiced x unit_price, USD", "never"),
    "payment_status": ("invoices", "feature", "Paid / Partially Paid / Unpaid as of 2026-09-30", "never"),
    "amount_paid": ("invoices", "feature", "Amount paid on this line, USD", "never"),
    "supplier_name": ("suppliers", "feature", "Fictional supplier name", "never"),
    "supplier_category": ("suppliers.category", "feature", "Supplier category", "never"),
    "supplier_country": ("suppliers.country", "feature", "Supplier country", "never"),
    "supplier_tier": ("suppliers", "feature", "Small / Medium / Large", "never"),
    "po_date": ("purchase_orders", "feature", "PO date", "never"),
    "item_description": ("purchase_orders", "feature", "Item description", "never"),
    "quantity_ordered": ("purchase_orders", "feature", "Original ordered quantity (before any amendment)", "never"),
    "po_unit_price": ("purchase_orders.unit_price", "feature", "Original PO unit price, USD (before any amendment)", "never"),
    "po_currency": ("purchase_orders.currency", "feature", "PO currency (USD)", "never"),
    "approved_amendment_id": ("purchase_orders", "feature", "Approved amendment applying to the PO line",
                              "no approved amendment on the PO line"),
    "amendment_id": ("po_amendments", "feature", "Amendment on the PO line, any status (incl. Rejected)",
                     "no amendment of any status on the PO line"),
    "amendment_date": ("po_amendments", "feature", "Amendment effective/decision date", "no amendment"),
    "amendment_type": ("po_amendments", "feature", "PRICE or QUANTITY", "no amendment"),
    "amendment_status": ("po_amendments.status", "feature", "Approved or Rejected", "no amendment"),
    "amendment_previous_unit_price": ("po_amendments.previous_unit_price", "feature", "Price before amendment, USD",
                                      "no amendment, or QUANTITY amendment"),
    "amendment_new_unit_price": ("po_amendments.new_unit_price", "feature", "Requested/approved new price, USD",
                                 "no amendment, or QUANTITY amendment"),
    "amendment_previous_quantity": ("po_amendments.previous_quantity", "feature", "Quantity before amendment",
                                    "no amendment, or PRICE amendment"),
    "amendment_new_quantity": ("po_amendments.new_quantity", "feature", "Approved new quantity",
                               "no amendment, or PRICE amendment"),
    "amendment_decided_by": ("po_amendments.decided_by", "feature", "Approver role", "no amendment"),
    "approved_unit_price": ("derived", "derived", "Applicable approved price at invoice_date: PO price, replaced by "
                            "an Approved PRICE amendment dated on/before invoice_date. USD", "never"),
    "approved_quantity": ("derived", "derived", "Ordered quantity after an Approved QUANTITY amendment", "never"),
    "receipt_count": ("goods_receipts (aggregated)", "feature", "Number of receipt lines for the PO line", "never"),
    "receipt_ids": ("goods_receipts (aggregated)", "feature", "';'-list of receipt_id in date order", "never"),
    "receipt_dates": ("goods_receipts (aggregated)", "feature", "';'-list of receipt_date, aligned with receipt_ids", "never"),
    "receipt_quantities": ("goods_receipts (aggregated)", "feature", "';'-list of quantity_received, aligned", "never"),
    "first_receipt_date": ("goods_receipts (aggregated)", "feature", "Earliest receipt date for the PO line", "never"),
    "last_receipt_date": ("goods_receipts (aggregated)", "feature", "Latest receipt date for the PO line", "never"),
    "quantity_received_total": ("goods_receipts (aggregated)", "feature", "Sum of ALL receipt lines for the PO line "
                                "(per PO line: do not sum across invoice rows)", "never"),
    "quantity_received_as_of_invoice_date": ("derived", "derived", "Receipts dated on/before invoice_date", "never"),
    "is_anomaly": ("ground_truth", "label", "1 anomaly / 0 normal", "never"),
    "primary_anomaly_type": ("ground_truth", "label", "NORMAL or one of four anomaly types", "never"),
    "all_applicable_anomaly_types": ("ground_truth", "label", "'|'-list, primary first; NORMAL for normal rows", "never"),
    "expected_exposure": ("ground_truth", "label", "Potential exposure, USD (duplicates: on repeat entry only)", "never"),
    "exposure_currency": ("ground_truth", "label", "USD", "never"),
    "expected_evidence_ids": ("ground_truth", "label", "';'-list of IDs supporting the case", "normal rows"),
    "case_description": ("ground_truth", "label", "Plain-language explanation", "never"),
    "exposure_basis": ("ground_truth", "label", "Formula used for expected_exposure", "never"),
    "exposure_paid_to_date": ("ground_truth", "label", "Realized part of the exposure (already paid), USD", "never"),
    "duplicate_group_id": ("ground_truth", "label", "Duplicate pair ID", "not a DUPLICATE_INVOICE row"),
    "hist_n_prior": ("ground_truth", "label", "Prior lines in the historical baseline", "not a HISTORICAL_PRICE_SPIKE row"),
    "hist_baseline_median": ("ground_truth", "label", "Median prior unit price, USD", "not a HISTORICAL_PRICE_SPIKE row"),
    "hist_price_ratio": ("ground_truth", "label", "unit_price / baseline median (4 dp)", "not a HISTORICAL_PRICE_SPIKE row"),
    "scenario_tags": ("ground_truth", "label", "';'-list of scenario / variant tags", "never (may be empty for plain normal rows)"),
}
LABEL_COLUMNS = [c for c, v in SCHEMA.items() if v[1] == "label"]


def cents(s):
    return (s.astype(float) * 100).round().astype("int64")


def load():
    t = {}
    t["inv"] = pd.read_csv(f"{DATA}/invoices.csv", dtype={"invoice_number": str}, parse_dates=["invoice_date"])
    t["po"] = pd.read_csv(f"{DATA}/purchase_orders.csv", dtype={"approved_amendment_id": str}, parse_dates=["po_date"])
    t["gr"] = pd.read_csv(f"{DATA}/goods_receipts.csv", parse_dates=["receipt_date"])
    t["amd"] = pd.read_csv(f"{DATA}/po_amendments.csv", parse_dates=["amendment_date"],
                           dtype={"previous_quantity": "Int64", "new_quantity": "Int64"})
    t["sup"] = pd.read_csv(f"{DATA}/suppliers.csv")
    t["gt"] = pd.read_csv(f"{DATA}/ground_truth.csv", keep_default_na=False,
                          na_values={c: [""] for c in ["hist_n_prior", "hist_baseline_median", "hist_price_ratio"]},
                          dtype={"hist_n_prior": "Int64"})
    return t


def merge_checked(left, right, on, name, conflicts):
    """Left join that refuses to duplicate or drop rows."""
    if right[on].duplicated().any():
        conflicts.append(f"{name}: join key {on} not unique on the right side")
        raise RuntimeError(conflicts[-1])
    out = left.merge(right, on=on, how="left", validate="many_to_one", indicator=True)
    missing = int((out["_merge"] != "both").sum())
    if missing or len(out) != len(left):
        conflicts.append(f"{name}: {missing} rows without a match")
        raise RuntimeError(conflicts[-1])
    return out.drop(columns="_merge")


def build(t, conflicts):
    inv, po, gr, amd, sup, gt = t["inv"], t["po"], t["gr"], t["amd"], t["sup"], t["gt"]

    # ground truth 1:1
    u = merge_checked(inv, gt.rename(columns={"invoice_id": "_gt_invoice_id"}), "invoice_line_id", "ground_truth", conflicts)
    if (u._gt_invoice_id != u.invoice_id).any():
        conflicts.append("ground_truth: invoice_id disagrees with invoices.csv")
    u = u.drop(columns="_gt_invoice_id")

    # suppliers N:1
    s = sup.rename(columns={"category": "supplier_category", "country": "supplier_country"})
    u = merge_checked(u, s, "supplier_id", "suppliers", conflicts)

    # purchase-order lines N:1 (shared keys must agree, then dropped)
    p = po.rename(columns={"unit_price": "po_unit_price", "currency": "po_currency", "po_id": "_po_id",
                           "supplier_id": "_po_supplier_id", "item_id": "_po_item_id"})
    u = merge_checked(u, p, "po_line_id", "purchase_orders", conflicts)
    for a, b in [("po_id", "_po_id"), ("supplier_id", "_po_supplier_id"), ("item_id", "_po_item_id")]:
        n = int((u[a] != u[b]).sum())
        if n:
            conflicts.append(f"purchase_orders: {n} rows where invoice {a} != PO {a}")
    u = u.drop(columns=["_po_id", "_po_supplier_id", "_po_item_id"])

    # amendments N:1 (optional: missing = no amendment)
    a = amd.rename(columns={"status": "amendment_status", "previous_unit_price": "amendment_previous_unit_price",
                            "new_unit_price": "amendment_new_unit_price", "previous_quantity": "amendment_previous_quantity",
                            "new_quantity": "amendment_new_quantity", "decided_by": "amendment_decided_by",
                            "po_id": "_amd_po_id"})
    if a.po_line_id.duplicated().any():
        conflicts.append("po_amendments: more than one amendment on a PO line")
        raise RuntimeError(conflicts[-1])
    u = u.merge(a, on="po_line_id", how="left", validate="many_to_one")
    bad = u._amd_po_id.notna() & (u._amd_po_id != u.po_id)
    if bad.any():
        conflicts.append(f"po_amendments: {int(bad.sum())} rows where amendment po_id != invoice po_id")
    u = u.drop(columns="_amd_po_id")

    # receipts aggregated per PO line (never row-joined)
    g = gr.sort_values(["receipt_date", "receipt_id"])
    if (g.po_line_id.map(po.set_index("po_line_id").item_id) != g.item_id).any():
        conflicts.append("goods_receipts: receipt item differs from PO line item")
    agg = g.groupby("po_line_id").agg(
        receipt_count=("receipt_id", "size"),
        receipt_ids=("receipt_id", lambda x: SEP.join(x)),
        receipt_dates=("receipt_date", lambda x: SEP.join(d.strftime("%Y-%m-%d") for d in x)),
        receipt_quantities=("quantity_received", lambda x: SEP.join(str(int(q)) for q in x)),
        first_receipt_date=("receipt_date", "min"),
        last_receipt_date=("receipt_date", "max"),
        quantity_received_total=("quantity_received", "sum")).reset_index()
    u = merge_checked(u, agg, "po_line_id", "goods_receipts(aggregated)", conflicts)
    rec = {p_: list(zip(x.receipt_date, x.quantity_received)) for p_, x in g.groupby("po_line_id")}
    u["quantity_received_as_of_invoice_date"] = [sum(q for d, q in rec[p_] if d <= dt)
                                                 for p_, dt in zip(u.po_line_id, u.invoice_date)]

    # derived: approved price at invoice date, approved quantity
    price_amd = (u.amendment_status == "Approved") & (u.amendment_type == "PRICE") & (u.amendment_date <= u.invoice_date)
    u["approved_unit_price"] = np.where(price_amd, u.amendment_new_unit_price, u.po_unit_price)
    qty_amd = (u.amendment_status == "Approved") & (u.amendment_type == "QUANTITY")
    u["approved_quantity"] = np.where(qty_amd, u.amendment_new_quantity.astype("Float64").fillna(0),
                                      u.quantity_ordered).astype("int64")
    return u[list(SCHEMA)]


# ----------------------------------------------------------------------------
# Validation (on the unified file as written and re-loaded)
# ----------------------------------------------------------------------------
def rules_from_unified(u):
    """The four reference rules recomputed from the unified file alone."""
    r = u.copy()
    r["price_c"], r["appr_c"] = cents(r.unit_price), cents(r.approved_unit_price)
    r["flag_price"] = r.price_c > r.appr_c
    r = r.sort_values(["invoice_date", "invoice_line_id"], kind="mergesort")
    r["cum_inv"] = r.groupby("po_line_id").quantity_invoiced.cumsum()
    r["excess_qty"] = np.minimum(r.quantity_invoiced, (r.cum_inv - r.quantity_received_total).clip(lower=0))
    r["flag_qty"] = r.excess_qty > 0
    docs = r.drop_duplicates("invoice_id")[["invoice_id", "supplier_id", "invoice_number", "invoice_date"]]
    docs = docs.sort_values(["invoice_date", "invoice_id"])
    k = [docs.supplier_id, docs.invoice_number]
    docs["rank"], docs["size"] = docs.groupby(k).cumcount(), docs.groupby(k).invoice_id.transform("count")
    dx = docs.set_index("invoice_id")
    r["dup_rank"], r["flag_dup"] = r.invoice_id.map(dx["rank"]), r.invoice_id.map(dx["size"]) > 1
    flag = {}
    for _, gp in r.groupby(["supplier_id", "item_id"]):
        d, p = gp.invoice_date.to_numpy(), gp.price_c.to_numpy()
        for j, ix in enumerate(gp.index):
            m = d < d[j]
            flag[ix] = bool(m.sum() >= HIST_MIN_PRIOR and p[j] > HIST_RATIO * np.median(p[m]))
    r["flag_hist"] = pd.Series(flag)
    return r.sort_index()


def validate(u, t, conflicts):
    C = []

    def chk(name, ok, detail=""):
        C.append((name, bool(ok), str(detail)))

    inv, po, gr, amd, sup, gt = (t[k] for k in ["inv", "po", "gr", "amd", "sup", "gt"])
    # schema & counts
    chk("Schema: columns and order match unified_schema.csv", list(u.columns) == list(SCHEMA), f"{u.shape[1]} columns")
    chk("Exactly 5,000 rows", len(u) == N_ROWS, len(u))
    chk("invoice_line_id unique (no duplicated records from joins)", u.invoice_line_id.is_unique)
    chk("No fully duplicated rows", not u.duplicated().any())
    chk("Same invoice lines, same order as invoices.csv", (u.invoice_line_id.values == inv.invoice_line_id.values).all())
    chk("4,500 normal / 500 anomalous (10.00%)", (u.is_anomaly == 0).sum() == N_NORMAL and (u.is_anomaly == 1).sum() == N_ANOM,
        f"normal={(u.is_anomaly == 0).sum()} anomalous={(u.is_anomaly == 1).sum()} rate={u.is_anomaly.mean():.2%}")
    vc = u.primary_anomaly_type.value_counts()
    chk("Primary type counts preserved", all(vc.get(k, 0) == v for k, v in TYPE_COUNTS.items()) and vc.get("NORMAL", 0) == N_NORMAL,
        ", ".join(f"{k}={vc.get(k, 0)}" for k in TYPE_COUNTS))
    chk("No join conflicts recorded", not conflicts, "; ".join(conflicts) or "none")

    # lossless round trip: every source table can be rebuilt exactly from the unified file
    inv_cols = list(inv.columns)
    chk("invoices.csv reproduced exactly", u[inv_cols].reset_index(drop=True).equals(inv[inv_cols]))
    gt_cols = list(gt.columns)
    chk("ground_truth.csv reproduced exactly", u[gt_cols].reset_index(drop=True).equals(gt[gt_cols]))
    s2 = u[["supplier_id", "supplier_name", "supplier_category", "supplier_country", "supplier_tier"]].drop_duplicates()
    s2 = s2.rename(columns={"supplier_category": "category", "supplier_country": "country"}).sort_values("supplier_id")
    chk("suppliers.csv reproduced exactly (80 suppliers)", s2.reset_index(drop=True).equals(sup.sort_values("supplier_id").reset_index(drop=True)))
    p2 = u[["po_id", "po_line_id", "supplier_id", "po_date", "item_id", "item_description", "quantity_ordered",
            "po_unit_price", "po_currency", "approved_amendment_id"]].drop_duplicates("po_line_id")
    p2 = p2.rename(columns={"po_unit_price": "unit_price", "po_currency": "currency"}).sort_values("po_line_id")
    chk("purchase_orders.csv reproduced exactly", p2.reset_index(drop=True).equals(po.sort_values("po_line_id").reset_index(drop=True)),
        f"{len(p2)} PO lines")
    a2 = u[u.amendment_id.notna()].drop_duplicates("amendment_id")
    a2 = pd.DataFrame({"amendment_id": a2.amendment_id, "po_id": a2.po_id, "po_line_id": a2.po_line_id,
                       "amendment_date": a2.amendment_date, "amendment_type": a2.amendment_type,
                       "status": a2.amendment_status, "previous_unit_price": a2.amendment_previous_unit_price,
                       "new_unit_price": a2.amendment_new_unit_price, "previous_quantity": a2.amendment_previous_quantity,
                       "new_quantity": a2.amendment_new_quantity, "decided_by": a2.amendment_decided_by}).sort_values("amendment_id")
    chk("po_amendments.csv reproduced exactly (incl. rejected)", a2.reset_index(drop=True).equals(amd.sort_values("amendment_id").reset_index(drop=True)),
        f"{len(a2)} amendments")
    pl = u.drop_duplicates("po_line_id")
    rows = []
    for _, x in pl.iterrows():
        for rid, d, q in zip(x.receipt_ids.split(SEP), x.receipt_dates.split(SEP), x.receipt_quantities.split(SEP)):
            rows.append((rid, x.po_id, x.po_line_id, pd.Timestamp(d), x.item_id, int(q)))
    g2 = pd.DataFrame(rows, columns=gr.columns).sort_values("receipt_id").reset_index(drop=True)
    chk("goods_receipts.csv reproduced exactly from receipt lists", g2.equals(gr.sort_values("receipt_id").reset_index(drop=True)
                                                                            .astype(g2.dtypes.to_dict())), f"{len(g2)} receipt lines")
    chk("Receipt aggregates consistent with lists", (pl.receipt_count == pl.receipt_ids.str.count(SEP) + 1).all()
        and (pl.quantity_received_total == pl.receipt_quantities.apply(lambda s: sum(map(int, s.split(SEP))))).all())
    chk("PO-line attributes identical on every row of the same PO line",
        u.groupby("po_line_id")[["po_date", "po_unit_price", "quantity_ordered", "quantity_received_total", "receipt_ids"]]
        .nunique(dropna=False).max().max() == 1)

    # relationships
    chk("No missing values in mandatory columns", not u[[c for c, v in SCHEMA.items() if v[3] == "never"
                                                          and c != "scenario_tags"]].isna().any().any())
    chk("po_line_id belongs to po_id", (u.po_line_id.str.rsplit("-", n=1).str[0] == u.po_id).all())
    chk("invoice_line_id belongs to invoice_id", (u.invoice_line_id.str.rsplit("-", n=1).str[0] == u.invoice_id).all())
    chk("approved_amendment_id set iff an Approved amendment exists",
        ((u.approved_amendment_id.notna()) == (u.amendment_status == "Approved")).all()
        and (u.approved_amendment_id.dropna() == u.amendment_id[u.approved_amendment_id.notna()]).all())
    chk("Each invoice document has one supplier, number, date, status and PO",
        u.groupby("invoice_id")[["supplier_id", "invoice_number", "invoice_date", "payment_status", "po_id"]].nunique().max().max() == 1)

    # financial consistency
    chk("Quantities positive", (u.quantity_invoiced > 0).all() and (u.quantity_ordered > 0).all()
        and (u.quantity_received_total > 0).all() and (u.approved_quantity > 0).all())
    chk("Prices positive", (u.unit_price > 0).all() and (u.po_unit_price > 0).all() and (u.approved_unit_price > 0).all())
    chk("Single currency USD", set(u.currency) == {"USD"} and set(u.po_currency) == {"USD"} and set(u.exposure_currency) == {"USD"})
    mon = [c for c in MONEY if c in u.columns]
    chk("All monetary columns have exactly two decimals", all(np.allclose(u[c].dropna() * 100, (u[c].dropna() * 100).round(), atol=1e-6) for c in mon),
        ", ".join(mon))
    chk("line_amount = quantity_invoiced x unit_price", (cents(u.line_amount) == u.quantity_invoiced * cents(u.unit_price)).all())
    a_, p_ = cents(u.line_amount), cents(u.amount_paid)
    chk("amount_paid consistent with payment_status",
        (p_[u.payment_status == "Paid"] == a_[u.payment_status == "Paid"]).all() and (p_[u.payment_status == "Unpaid"] == 0).all()
        and ((p_ > 0) & (p_ < a_))[u.payment_status == "Partially Paid"].all())
    chk("Dates plausible: PO <= first receipt <= last receipt; PO <= invoice; first receipt of PO <= invoice",
        (u.po_date <= u.first_receipt_date).all() and (u.first_receipt_date <= u.last_receipt_date).all()
        and (u.po_date <= u.invoice_date).all()
        and (u.invoice_date >= u.po_id.map(u.groupby("po_id").first_receipt_date.min())).all())
    has_a = u.amendment_id.notna()
    chk("Amendment dated between PO date and the PO line's first invoice",
        ((u.amendment_date[has_a] >= u.po_date[has_a]) &
         (u.amendment_date[has_a] <= u.po_line_id.map(u.groupby("po_line_id").invoice_date.min())[has_a])).all())
    chk("approved_quantity >= quantity_ordered; > only with an approved quantity amendment",
        (u.approved_quantity >= u.quantity_ordered).all()
        and ((u.approved_quantity > u.quantity_ordered) == ((u.amendment_status == "Approved") & (u.amendment_type == "QUANTITY"))).all())
    rej = u.amendment_status == "Rejected"
    chk("Rejected amendments never change the approved price", (cents(u.approved_unit_price[rej]) == cents(u.po_unit_price[rej])).all(),
        f"{int(rej.sum())} rows with a rejected amendment")

    # intentional missing values
    def miss_ok(col, expected_missing):
        return (u[col].isna() == expected_missing).all()
    no_amd = u.amendment_id.isna()
    is_q, is_p = u.amendment_type == "QUANTITY", u.amendment_type == "PRICE"
    is_dup, is_hist = u.primary_anomaly_type == "DUPLICATE_INVOICE", u.primary_anomaly_type == "HISTORICAL_PRICE_SPIKE"
    checks_missing = [miss_ok("approved_amendment_id", u.amendment_status != "Approved"),
                      all(miss_ok(c, no_amd) for c in ["amendment_date", "amendment_type", "amendment_status", "amendment_decided_by"]),
                      miss_ok("amendment_previous_unit_price", no_amd | is_q), miss_ok("amendment_new_unit_price", no_amd | is_q),
                      miss_ok("amendment_previous_quantity", no_amd | is_p), miss_ok("amendment_new_quantity", no_amd | is_p),
                      all(miss_ok(c, ~is_hist) for c in ["hist_n_prior", "hist_baseline_median", "hist_price_ratio"]),
                      ((u.duplicate_group_id == "") == ~is_dup).all(),
                      ((u.expected_evidence_ids == "") == (u.is_anomaly == 0)).all()]
    chk("Every missing value matches its documented rule (no arbitrary gaps)", all(checks_missing),
        "; ".join(f"{c}={int(u[c].isna().sum())}" for c in u.columns if u[c].isna().any()))

    # ground truth correctness, recomputed from the unified file alone
    r = rules_from_unified(u)
    flags = {"DUPLICATE_INVOICE": "flag_dup", "QUANTITY_DISCREPANCY": "flag_qty",
             "PO_PRICE_MISMATCH": "flag_price", "HISTORICAL_PRICE_SPIKE": "flag_hist"}
    for t_, f in flags.items():
        sel = r.primary_anomaly_type == t_
        chk(f"{t_}: all planted rows fire the rule on the unified file", r.loc[sel, f].all(), f"{int(r.loc[sel, f].sum())}/{int(sel.sum())}")
    anyf = r[list(flags.values())].any(axis=1)
    chk("No normal row fires any rule (no unintended anomalies after the join)", not anyf[r.is_anomaly == 0].any(),
        f"{int(anyf[r.is_anomaly == 0].sum())} violations")
    fired = r.apply(lambda x: {t_ for t_, f in flags.items() if x[f]}, axis=1)
    labelled = r.all_applicable_anomaly_types.str.split("|").apply(lambda l: set(l) - {"NORMAL"})
    chk("all_applicable_anomaly_types equals the set of rules that fire (no contradictory labels)", (fired == labelled).all(),
        f"{int((fired != labelled).sum())} mismatches")
    chk("Primary type listed first in all_applicable_anomaly_types",
        (r.all_applicable_anomaly_types.str.split("|").str[0] == r.primary_anomaly_type).all())
    dk = r[r.flag_dup].drop_duplicates("invoice_id").groupby(["supplier_id", "invoice_number"]).size()
    chk("Duplicates: 40 pairs by key (supplier_id, invoice_number), all labelled", len(dk) == 40 and (dk == 2).all()
        and (r.loc[r.flag_dup, "primary_anomaly_type"] == "DUPLICATE_INVOICE").all())

    # exposure, recomputed
    exp = np.where(r.primary_anomaly_type == "NORMAL", 0,
          np.where(r.primary_anomaly_type == "DUPLICATE_INVOICE", np.where(r.dup_rank > 0, a_, 0),
          np.where(r.primary_anomaly_type == "QUANTITY_DISCREPANCY", r.excess_qty * r.appr_c,
                   (r.price_c - r.appr_c).clip(lower=0) * r.quantity_invoiced)))
    mism = int((cents(r.expected_exposure).values != exp).sum())
    chk("expected_exposure matches the formulas recomputed from the unified file", mism == 0,
        f"{mism} mismatches; total ${r.expected_exposure.sum():,.2f}")
    d = r[r.primary_anomaly_type == "DUPLICATE_INVOICE"]
    per_pair = d.groupby("duplicate_group_id").expected_exposure.apply(lambda s: (s > 0).sum())
    chk("Duplicate potential exposure counted once per pair", (per_pair == 1).all(), f"{len(per_pair)} pairs")
    paid = d.groupby("duplicate_group_id").apply(lambda x: max(0, cents(x.amount_paid).sum() - cents(x.line_amount).iloc[0]),
                                                 include_groups=False)
    real = d.groupby("duplicate_group_id").exposure_paid_to_date.apply(lambda s: int(round(s.sum() * 100)))
    chk("Duplicate realized loss = max(0, paid on both entries - line amount)", (paid.sort_index().values == real.sort_index().values).all(),
        f"potential ${d.expected_exposure.sum():,.2f} vs realized ${d.exposure_paid_to_date.sum():,.2f}")
    chk("Realized exposure never exceeds potential exposure", (r.exposure_paid_to_date <= r.expected_exposure + 1e-9).all())
    h = r[r.primary_anomaly_type == "HISTORICAL_PRICE_SPIKE"]
    nb = h.scenario_tags.str.contains("NO_CONTRACT_BASIS")
    chk("Historical spikes without a contractual reference carry zero exposure",
        (h.expected_exposure[nb] == 0).all() and (h.expected_exposure[~nb] > 0).all() and (~h.flag_price[nb]).all(),
        f"{int(nb.sum())} zero, {int((~nb).sum())} PO-based")
    chk("Normal rows: exposure 0, all_applicable NORMAL", (r.expected_exposure[r.is_anomaly == 0] == 0).all()
        and (r.all_applicable_anomaly_types[r.is_anomaly == 0] == "NORMAL").all())

    # normal edge cases survive the join
    n = r[r.is_anomaly == 0]
    chk("Normal partial deliveries later completed remain normal",
        ((n.quantity_received_as_of_invoice_date < n.cum_inv) & ~n.flag_qty).sum() > 0,
        f"{int(((n.quantity_received_as_of_invoice_date < n.cum_inv)).sum())} rows")
    chk("Normal approved price changes remain normal", ((n.unit_price > n.po_unit_price) & ~n.flag_price).sum() > 0,
        f"{int((n.unit_price > n.po_unit_price).sum())} rows")
    return C


def main():
    t = load()
    conflicts = []
    u = build(t, conflicts)
    out = u.copy()
    for c in MONEY:
        out[c] = out[c].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    for c in DATES:
        out[c] = out[c].dt.strftime("%Y-%m-%d")
    out.to_csv(OUT, index=False)
    pd.DataFrame([dict(column=c, source=v[0], role=v[1], description=v[2], missing_when=v[3])
                  for c, v in SCHEMA.items()]).to_csv(os.path.join(DATA, "unified_schema.csv"), index=False)

    # re-load exactly as a downstream pipeline would
    u2 = pd.read_csv(OUT, parse_dates=DATES, keep_default_na=False,
                     na_values={c: [""] for c in SCHEMA if SCHEMA[c][3] != "never" and c not in
                                ("expected_evidence_ids", "duplicate_group_id", "scenario_tags")},
                     dtype={"invoice_number": str, "approved_amendment_id": str, "amendment_previous_quantity": "Int64",
                            "amendment_new_quantity": "Int64", "hist_n_prior": "Int64", "scenario_tags": str})
    C = validate(u2, t, conflicts)
    passed = sum(ok for _, ok, _ in C)
    L = ["# Unified dataset validation report", "", f"Output: `data/ai_leak_hunter_unified.csv` ({len(u2):,} rows x {u2.shape[1]} columns)", "",
         "## Join summary", "", "| Source | Rows | Cardinality to invoice line | Join key | Unmatched / lost |", "|---|---|---|---|---|",
         f"| invoices.csv | {len(t['inv']):,} | base | invoice_line_id | 0 |",
         f"| ground_truth.csv | {len(t['gt']):,} | 1:1 | invoice_line_id | 0 |",
         f"| suppliers.csv | {len(t['sup']):,} | N:1 | supplier_id | 0 |",
         f"| purchase_orders.csv | {len(t['po']):,} | N:1 | po_line_id | 0 (every PO line is invoiced) |",
         f"| po_amendments.csv | {len(t['amd']):,} | N:1, optional | po_line_id | 0 |",
         f"| goods_receipts.csv | {len(t['gr']):,} | aggregated per po_line_id, then N:1 | po_line_id | 0 |", "",
         f"Conflicts: {'; '.join(conflicts) or 'none'}.", "", "## Checks", "", "| # | Check | Result | Detail |", "|---|---|---|---|"]
    for i, (nme, ok, det) in enumerate(C, 1):
        L.append(f"| {i} | {nme} | {'PASS' if ok else '**FAIL**'} | {det.replace('|', '/')} |")
    L += ["", f"**{passed}/{len(C)} checks passed.**", "", "## Class counts", "", "| primary_anomaly_type | Rows | Share |", "|---|---|---|"]
    for k, v in u2.primary_anomaly_type.value_counts().items():
        L.append(f"| {k} | {v} | {v / len(u2):.1%} |")
    L += ["", "Label columns (drop before detection): " + ", ".join(f"`{c}`" for c in LABEL_COLUMNS)]
    open(os.path.join(DATA, "unified_validation_report.md"), "w").write("\n".join(L) + "\n")
    w = max(len(x) for x, _, _ in C)
    for nme, ok, det in C:
        print(f"  {'PASS' if ok else 'FAIL'}  {nme:<{w}}  {det}")
    print(f"\n{passed}/{len(C)} checks passed. Wrote {OUT}")
    if passed != len(C):
        sys.exit("Unified dataset validation failed.")


if __name__ == "__main__":
    main()
