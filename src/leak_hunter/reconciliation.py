"""Three-way reconciliation: one enriched row per invoice line.

For each invoice line this computes, from procurement data only:

approved_unit_price      PO unit price, replaced by the latest APPROVED PRICE
                         amendment dated on/before the invoice date. Rejected
                         (or any non-approved) amendments are never applied.
approved_quantity        PO quantity, replaced by the latest APPROVED QUANTITY
                         amendment (quantity authorisation is not date-bound).
received_total           sum of ALL receipt lines of the PO line, including
                         receipts dated after the invoice (reconciliation is as
                         of the data extraction).
received_as_of_invoice   receipts dated on/before the invoice date (context, and
                         the Isolation Forest's point-in-time quantity feature).
cum_invoiced             running invoiced quantity on the PO line, ordered by
                         (invoice_date, invoice_id, invoice_line_id).
supported_quantity       min(received_total, approved_quantity)
excess_qty               units of this line beyond supported_quantity:
                         max(0, cum - S) - max(0, cum_prev - S)
excess_over_approved     units of this line beyond the approved quantity alone
                         (no future receipt can justify these)
invoice_age_days         as_of_date - invoice_date (as_of_date: configured, else
                         the latest invoice/receipt date in the data)
dup_group_key / dup_rank duplicate-invoice groups (see ``duplicate_groups``);
                         rank 0 = original (earliest), >0 = re-entries.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import DetectionConfig
from .data_loader import ProcurementData
from .exposure import from_cents, to_cents


def _tokens(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [t for t in re.split(r"[^0-9A-Za-z]+", str(value).upper()) if t]


def _strip_zeros(token: str) -> str:
    # leading zeros of every digit run inside ONE token ("A0012" -> "A12", "000" -> "0")
    return re.sub(r"(?<![0-9])0+(?=[0-9])", "", token)


def normalize_invoice_number(value, strip_leading_zeros: bool = False) -> str:
    """Exact matching key: upper-case, letters and digits only.

    "inv-2026/000123 " -> "INV2026000123". With ``strip_leading_zeros`` the
    leading zeros of each digit run are removed *per separator-delimited token*
    before the tokens are joined ("SI/26/012345" -> "SI2612345"). (v1 removed
    separators first, which merged "26" and "012345" into one run, so inner
    zeros were never stripped.)
    """
    toks = _tokens(value)
    if strip_leading_zeros:
        toks = [_strip_zeros(t) for t in toks]
    return "".join(toks)


OCR_SUBSTITUTIONS = {"O": "0", "Q": "0", "I": "1", "L": "1"}


def fuzzy_invoice_key(value, substitutions: dict[str, str] | None = None, strip_zeros: bool = True) -> str:
    """Looser key for fuzzy matching: look-alike characters mapped to digits
    (O/Q->0, I/L->1), then (by default) leading zeros stripped per token, then
    joined. Equality of this key is a match; edit distance is measured on the
    key WITHOUT zero stripping, so "0001" vs "0011" stays a digit substitution."""
    subs = OCR_SUBSTITUTIONS if substitutions is None else substitutions
    toks = ["".join(subs.get(c, c) for c in t) for t in _tokens(value)]
    return "".join(_strip_zeros(t) if strip_zeros else t for t in toks)


def is_digit_substitution(a: str, b: str) -> bool:
    """True when equal-length keys differ only by digit-for-digit substitutions
    (and not by an adjacent swap) - the signature of a DIFFERENT, sequentially
    numbered invoice (INV-1041 vs INV-1042 / INV-1052), not of a re-keyed duplicate."""
    if len(a) != len(b):
        return False
    diff = [i for i in range(len(a)) if a[i] != b[i]]
    if not diff or not all(a[i].isdigit() and b[i].isdigit() for i in diff):
        return False
    swap = len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]
    return not swap


def osa_distance(a: str, b: str, limit: int = 1) -> int:
    """Optimal-string-alignment (restricted Damerau-Levenshtein) distance:
    insertions, deletions, substitutions and adjacent transpositions each cost 1.
    Returns ``limit + 1`` early when the distance must exceed ``limit``."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    la, lb = len(a), len(b)
    prev2, prev = None, list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[lb]


def duplicate_groups(docs: pd.DataFrame, rules) -> pd.DataFrame:
    """Group invoice documents that are entries of the same supplier invoice.

    ``docs``: one row per invoice_id with supplier_id, invoice_number,
    invoice_date, doc_amount_c (cents) and po_lines (sorted tuple).

    Two documents of the SAME supplier are linked when
      EXACT  their exact keys are equal (any amount, any date), or
      FUZZY  (rules.dup_fuzzy enabled) their amounts are identical (within
             rules.dup_fuzzy_amount_tolerance), their invoice dates are at most
             rules.dup_fuzzy_window_days apart, and their fuzzy keys are equal
             (look-alikes, leading zeros, separators) or their unstripped fuzzy
             keys are within OSA distance rules.dup_fuzzy_max_distance (keys of
             at least rules.dup_fuzzy_min_length characters). A single
             digit-for-digit substitution is NOT a fuzzy match unless
             rules.dup_fuzzy_allow_digit_substitution: it is indistinguishable
             from the next invoice in a sequence (recurring fixed-amount
             billing); such re-keyed duplicates are left to Rule B, which sees
             the extra quantity against receipts.
    Linked documents form groups (connected components). Returns docs with
    dup_group_key, dup_rank, dup_group_size, dup_match_type, dup_edit_distance.
    """
    d = docs.copy()
    d["exact_key"] = d.invoice_number.map(lambda s: normalize_invoice_number(s, rules.invoice_number_strip_leading_zeros))
    d["fuzzy_key"] = d.invoice_number.map(fuzzy_invoice_key)
    d["edit_key"] = d.invoice_number.map(lambda s: fuzzy_invoice_key(s, strip_zeros=False))
    d = d.sort_values(["invoice_date", "invoice_id"]).reset_index(drop=True)
    parent = list(range(len(d)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edges = []  # (i, j, type, distance)
    valid = d.exact_key != ""
    for _, g in d[valid].groupby(["supplier_id", "exact_key"]):
        idx = g.index.tolist()
        edges += [(idx[0], j, "EXACT", 0) for j in idx[1:]]
    if rules.dup_fuzzy:
        tol = int(round(rules.dup_fuzzy_amount_tolerance * 100))
        win = rules.dup_fuzzy_window_days
        for _, g in d[valid].groupby("supplier_id"):
            g = g.sort_values("doc_amount_c")
            rows = list(g.itertuples())
            for a in range(len(rows)):
                for b in range(a + 1, len(rows)):
                    ra, rb = rows[a], rows[b]
                    if rb.doc_amount_c - ra.doc_amount_c > tol:
                        break
                    if ra.exact_key == rb.exact_key:
                        continue
                    gap = abs((rb.invoice_date - ra.invoice_date).days)
                    if gap > win:
                        continue
                    if rules.dup_recurring_interval_days and gap == rules.dup_recurring_interval_days:
                        continue          # recurring subscription / retainer cadence, not a duplicate
                    if rules.dup_fuzzy_require_same_items and ra.item_descs != rb.item_descs:
                        continue
                    if ra.fuzzy_key == rb.fuzzy_key:
                        dist = 0
                    elif min(len(ra.edit_key), len(rb.edit_key)) >= rules.dup_fuzzy_min_length:
                        dist = osa_distance(ra.edit_key, rb.edit_key, rules.dup_fuzzy_max_distance)
                        if dist > rules.dup_fuzzy_max_distance:
                            continue
                        if (not rules.dup_fuzzy_allow_digit_substitution
                                and is_digit_substitution(ra.edit_key, rb.edit_key)):
                            continue
                    else:
                        continue
                    edges.append((ra.Index, rb.Index, "FUZZY", dist))
    for i, j, _, _ in edges:
        parent[find(i)] = find(j)
    d["_root"] = [find(i) for i in range(len(d))]
    d["dup_group_size"] = d.groupby("_root").invoice_id.transform("size")
    d["dup_rank"] = d.groupby("_root").cumcount()
    first = d.groupby("_root").first()
    d["dup_group_key"] = np.where(d.dup_group_size > 1,
                                  d._root.map(first.supplier_id) + "|" + d._root.map(first.exact_key), None)
    fuzzy_roots = {find(i) for i, j, t, _ in edges if t == "FUZZY"}
    d["dup_match_type"] = np.where(d.dup_group_size > 1,
                                   np.where(d._root.isin(fuzzy_roots), "FUZZY", "EXACT"), None)
    dist = {}
    for i, j, t, k in edges:
        dist[j] = max(dist.get(j, 0), k)
        dist[i] = max(dist.get(i, 0), k)
    d["dup_edit_distance"] = [dist.get(i, 0) for i in range(len(d))]
    return d.drop(columns=["_root", "edit_key"])


def _status(s: pd.Series) -> pd.Series:
    return s.astype("object").fillna("").astype(str).str.strip().str.casefold()


def build_line_view(data: ProcurementData, config: DetectionConfig | None = None) -> pd.DataFrame:
    cfg = (config or DetectionConfig()).rules
    inv, po, am, gr = data.invoices, data.purchase_orders, data.po_amendments, data.goods_receipts
    v = inv.copy()
    computed = from_cents(to_cents((v.quantity_invoiced * v.unit_price).to_numpy()))
    if "line_amount" in v.columns:
        off = (to_cents(v.line_amount.to_numpy()) != to_cents(computed))
        if off.any():
            data.quality_issues.append({"check": "line_amount != quantity_invoiced x unit_price (recomputed)",
                                        "count": int(off.sum()), "examples": v.invoice_line_id[off].head(5).tolist()})
    v["line_amount"] = computed

    p = po.rename(columns={"unit_price": "po_unit_price", "currency": "po_currency", "supplier_id": "po_supplier_id",
                           "item_id": "po_item_id", "po_id": "po_po_id"})
    v = v.merge(p[["po_line_id", "po_po_id", "po_supplier_id", "po_item_id", "po_date", "item_description",
                   "quantity_ordered", "po_unit_price", "po_currency", "approved_amendment_id"]],
                on="po_line_id", how="left", validate="many_to_one")
    v["po_line_found"] = v.po_po_id.notna()

    # --- price authorisation at invoice date ------------------------------------------------
    st, tp = _status(am.status), _status(am.amendment_type)
    ap = am[(st == "approved") & (tp == "price") & am.new_unit_price.notna()]
    m = v[["invoice_line_id", "po_line_id", "invoice_date"]].merge(
        ap[["po_line_id", "amendment_id", "amendment_date", "new_unit_price"]], on="po_line_id", how="inner")
    applied = (m[m.amendment_date <= m.invoice_date].sort_values(["amendment_date", "amendment_id"])
               .drop_duplicates("invoice_line_id", keep="last").set_index("invoice_line_id"))
    later_m = m[m.amendment_date > m.invoice_date].sort_values(["amendment_date", "amendment_id"])
    later = later_m.groupby("invoice_line_id").amendment_id.apply(list)
    nxt = later_m.drop_duplicates("invoice_line_id").set_index("invoice_line_id")
    v["next_approved_amendment_date"] = pd.to_datetime(v.invoice_line_id.map(nxt.amendment_date.to_dict()))
    v["next_approved_amendment_price"] = v.invoice_line_id.map(nxt.new_unit_price.to_dict()).astype(float)
    v["next_approved_amendment_id"] = v.invoice_line_id.map(nxt.amendment_id.to_dict())
    v["applied_price_amendment_id"] = v.invoice_line_id.map(applied.amendment_id)
    v["approved_unit_price"] = v.invoice_line_id.map(applied.new_unit_price).fillna(v.po_unit_price)
    v["approved_price_amendments_after_invoice"] = v.invoice_line_id.map(later).apply(
        lambda x: x if isinstance(x, list) else [])
    na = am[(st != "approved") & (tp == "price")]
    rej: dict[str, list] = {}
    for pl, aid, sts, price in zip(na.po_line_id, na.amendment_id, _status(na.status), na.new_unit_price):
        rej.setdefault(pl, []).append((aid, sts, price))
    v["unapproved_price_amendments"] = v.po_line_id.map(lambda k: list(rej.get(k, [])))
    all_ids: dict[str, list] = {}
    for pl, aid in zip(am.po_line_id, am.amendment_id):
        all_ids.setdefault(pl, []).append(aid)
    v["all_amendment_ids"] = v.po_line_id.map(lambda k: list(all_ids.get(k, [])))

    # --- quantity authorisation -----------------------------------------------------------------
    aq = am[(st == "approved") & (tp == "quantity") & am.new_quantity.notna()].sort_values(
        ["amendment_date", "amendment_id"]).drop_duplicates("po_line_id", keep="last").set_index("po_line_id")
    v["applied_qty_amendment_id"] = v.po_line_id.map(aq.amendment_id)
    v["approved_quantity"] = v.po_line_id.map(aq.new_quantity).fillna(v.quantity_ordered)

    # --- receipts ---------------------------------------------------------------------------------
    g = gr.sort_values(["receipt_date", "receipt_id"])
    rec = {k: list(zip(x.receipt_id, x.receipt_date, x.quantity_received)) for k, x in g.groupby("po_line_id")}
    v["receipt_ids"] = v.po_line_id.map(lambda k: [r[0] for r in rec.get(k, [])])
    v["receipt_dates"] = v.po_line_id.map(lambda k: [r[1].strftime("%Y-%m-%d") for r in rec.get(k, [])])
    v["receipt_quantities"] = v.po_line_id.map(lambda k: [float(r[2]) for r in rec.get(k, [])])
    v["received_total"] = v.po_line_id.map(lambda k: float(sum(r[2] for r in rec.get(k, []))))
    v["received_as_of_invoice"] = [float(sum(r[2] for r in rec.get(k, []) if r[1] <= d))
                                   for k, d in zip(v.po_line_id, v.invoice_date)]
    v["receipts_after_invoice"] = [[r[0] for r in rec.get(k, []) if r[1] > d]
                                   for k, d in zip(v.po_line_id, v.invoice_date)]

    # --- cumulative invoicing per PO line -------------------------------------------------------
    order = v.sort_values(["po_line_id", "invoice_date", "invoice_id", "invoice_line_id"], kind="mergesort")
    v["cum_invoiced"] = order.groupby("po_line_id").quantity_invoiced.cumsum().reindex(v.index)
    v["previously_invoiced"] = v.cum_invoiced - v.quantity_invoiced
    v["invoice_lines_on_po_line"] = v.po_line_id.map(order.groupby("po_line_id").invoice_line_id.apply(list))
    v["supported_quantity"] = np.minimum(v.received_total, v.approved_quantity)

    def _excess(limit):
        now = (v.cum_invoiced - limit).clip(lower=0)
        before = (v.previously_invoiced - limit).clip(lower=0)
        return (now - before).where(v.po_line_found, 0.0)

    v["excess_qty"] = _excess(v.supported_quantity)
    v["excess_over_approved"] = _excess(v.approved_quantity)
    v["qty_limit_reason"] = np.where(v.received_total <= v.approved_quantity, "EXCEEDS_RECEIPTS",
                                     "EXCEEDS_APPROVED_QUANTITY")
    as_of = (pd.Timestamp(cfg.as_of_date) if cfg.as_of_date else
             max(inv.invoice_date.max(), gr.receipt_date.max() if len(gr) else inv.invoice_date.max()))
    v["as_of_date"] = as_of
    v["invoice_age_days"] = (as_of - v.invoice_date).dt.days

    # --- duplicate invoice-number groups (document level) ---------------------------------------
    v["invoice_number_norm"] = v.invoice_number.map(
        lambda s: normalize_invoice_number(s, cfg.invoice_number_strip_leading_zeros))
    v["invoice_number_fuzzy"] = v.invoice_number.map(fuzzy_invoice_key)
    docs = v.groupby("invoice_id", sort=False).agg(
        supplier_id=("supplier_id", "first"), invoice_number=("invoice_number", "first"),
        invoice_date=("invoice_date", "first"), doc_amount=("line_amount", "sum"),
        po_lines=("po_line_id", lambda s: tuple(sorted(set(s.dropna())))),
        item_descs=("item_description", lambda s: tuple(sorted(set(s.dropna().astype(str).str.strip().str.casefold()))))
    ).reset_index()
    docs["doc_amount_c"] = to_cents(docs.doc_amount.to_numpy())
    dx = duplicate_groups(docs, cfg).set_index("invoice_id")
    v["doc_amount"] = v.invoice_id.map(dx.doc_amount)
    v["dup_group_key"] = v.invoice_id.map(dx.dup_group_key)
    v["dup_rank"] = v.invoice_id.map(dx.dup_rank).fillna(0).astype(int)
    v["dup_group_size"] = v.invoice_id.map(dx.dup_group_size).fillna(1).astype(int)
    v["dup_match_type"] = v.invoice_id.map(dx.dup_match_type)
    v["dup_edit_distance"] = v.invoice_id.map(dx.dup_edit_distance).fillna(0).astype(int)
    v["lines_on_invoice"] = v.invoice_id.map(v.groupby("invoice_id").size())
    return v
