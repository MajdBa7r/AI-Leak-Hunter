"""The four deterministic detection rules.

Each rule takes the reconciled line view (``reconciliation.build_line_view``)
and returns a *hit table*: one row per invoice line it flags, with columns
``HIT_COLUMNS``. ``findings.build_findings`` turns hits into the common
findings schema.

Rule A  DUPLICATE_INVOICE
    Documents of the same supplier that are entries of the same invoice
    (``reconciliation.duplicate_groups``):
      EXACT  normalised numbers equal (upper-case, letters and digits only);
      FUZZY  identical document amount, invoice dates <= 60 days apart, and
             numbers equal after mapping look-alikes (O/Q->0, I/L->1) and
             stripping leading zeros, or within one edit (insert, delete,
             substitute, adjacent swap).
    The earliest document (invoice_date, then invoice_id) is the ORIGINAL; later
    ones are REPEAT entries. Every line of every document in the group is a hit.
    potential_exposure (REPEAT line) = line_amount of the repeat line
    potential_exposure (ORIGINAL line) = 0           (counted once per incident)
    confirmed_loss / paid_exposure (REPEAT) = realized duplicate payment
        = max(0, paid on original + paid on repeat(s) - original line amount),
          allocated in order and capped at the repeat line amount.
    A duplicate number is not a duplicate payment: confirmed_loss is 0 until
    payment records show money paid beyond the original amount.

Rule B  QUANTITY_DISCREPANCY
    Per PO line, invoice lines are processed in (invoice_date, invoice_id,
    invoice_line_id) order and their quantities accumulated.
        supported = min(sum of ALL receipt lines incl. later receipts,
                        approved quantity after approved quantity amendments)
        excess_i  = max(0, cum_i - supported) - max(0, cum_(i-1) - supported)
    A line is a hit when excess_i > 0 and the cumulative over-billing exceeds
    ``qty_tolerance_units``. Partial deliveries completed later are therefore
    not flagged, and invoices sharing a PO line are never compared one by one
    against the full receipt total.
    Grace period: while invoice age < qty_grace_days, receipts may still be
    posted, so only units beyond the APPROVED quantity are flagged; the rest
    is returned by ``pending_quantity_checks`` (status PENDING_RECEIPT).
    potential_exposure = flagged excess units x approved unit price.

Rule B materiality: a line is only flagged when the cumulative excess is at
    least qty_materiality_pct (1.5%) of the cumulative invoiced quantity AND
    the flagged units are worth >= qty_materiality_value ($25) at the approved
    price. Goods in transit: receipts dated after the invoice are counted (all
    receipts posted by the as-of date), and recent invoices fall under the grace
    period, so an invoice arriving shortly before its goods is never flagged.

Rule C  PO_PRICE_MISMATCH
    approved price = PO unit price, replaced by the latest APPROVED PRICE
    amendment dated on/before the invoice date (rejected or pending ones are
    ignored). Materiality (dual threshold): hit only when the line overcharge
    (unit difference x quantity) > price_materiality_abs ($5.00) AND the unit
    variance > price_materiality_pct (2%) of the approved price.
    Confidence: HIGH by default; MEDIUM when the line is billed at the price of
    a pending/draft amendment; LOW (BORDERLINE_AMENDMENT_TIMING) when an
    approved amendment covering the invoiced price is dated within
    amendment_borderline_business_days business days after the invoice.
    potential_exposure = (invoice unit price - approved price) x quantity_invoiced.

Rule D  HISTORICAL_PRICE_SPIKE
    Baseline = median (configurable) unit price of >= hist_min_prior valid
    invoice lines for the same supplier and item dated strictly earlier.
    Hit when unit price > effective threshold x baseline, where
    effective threshold = max(hist_ratio_threshold, 1 + k x robust CV) widens the
    band for naturally variable price histories.
    An unusual increase is not by itself a loss:
    potential_exposure = contractual excess (invoice price - approved PO price)
    x quantity when the invoice also exceeds the approved price by more than
    the Rule C tolerance, else 0.

paid_exposure for rules B-D = potential_exposure x amount_paid / line_amount.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DetectionConfig
from .exposure import DUP, HIST, PRICE, QTY, from_cents, prorate_paid, realized_duplicate_payments, to_cents

HIT_COLUMNS = ["invoice_line_id", "finding_type", "rule_id", "potential_exposure", "paid_exposure",
               "confirmed_loss", "exposure_basis", "evidence", "details", "description", "incident_key"]
RULE_IDS = {DUP: "RULE_A_DUPLICATE_INVOICE", QTY: "RULE_B_QUANTITY_DISCREPANCY",
            PRICE: "RULE_C_PO_PRICE_MISMATCH", HIST: "RULE_D_HISTORICAL_PRICE_SPIKE"}


def _usd(x: float) -> str:
    return f"${x:,.2f}"


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=HIT_COLUMNS)


def price_allowance_c(approved_c: np.ndarray, rules) -> np.ndarray:
    """Per-unit tolerance in cents: max(price_tolerance_abs, price_tolerance_pct x approved)."""
    return np.maximum(to_cents(rules.price_tolerance_abs), np.round(np.asarray(approved_c) * rules.price_tolerance_pct))


# ------------------------------------------------------------------------------------------------
# Rule A
# ------------------------------------------------------------------------------------------------
def rule_duplicate_invoices(view: pd.DataFrame, config: DetectionConfig | None = None) -> pd.DataFrame:
    v = view[view.dup_group_key.notna()]
    if v.empty:
        return _empty()
    rows = []
    for key, g in v.groupby("dup_group_key", sort=True):
        g = g.sort_values(["dup_rank", "invoice_line_id"])
        docs = g.drop_duplicates("invoice_id")
        orig_id = docs.invoice_id.iloc[0]
        orig = g[g.invoice_id == orig_id]
        doc_info = [{"invoice_id": r.invoice_id, "invoice_number": r.invoice_number,
                     "invoice_date": r.invoice_date.strftime("%Y-%m-%d"),
                     "document_amount": round(float(g.loc[g.invoice_id == r.invoice_id, "line_amount"].sum()), 2),
                     "amount_paid": round(float(g.loc[g.invoice_id == r.invoice_id, "amount_paid"].sum()), 2),
                     "payment_status": r.payment_status, "role": "ORIGINAL" if r.dup_rank == 0 else "REPEAT"}
                    for r in docs.itertuples()]
        # match each repeat line to an original line (same PO line, else same item)
        match = {}
        for r in g[g.dup_rank > 0].itertuples():
            cand = orig[orig.po_line_id == r.po_line_id]
            if cand.empty:
                cand = orig[orig.item_id == r.item_id]
            match[r.invoice_line_id] = cand.invoice_line_id.iloc[0] if len(cand) else None
        realized = {}
        for o_line, reps in pd.Series(match).dropna().groupby(lambda k: match[k]):
            o = orig[orig.invoice_line_id == o_line].iloc[0]
            rl = g[g.invoice_line_id.isin(reps.index)].sort_values(["dup_rank", "invoice_line_id"])
            res = realized_duplicate_payments(to_cents(o.line_amount),
                                              [to_cents(o.amount_paid)] + list(to_cents(rl.amount_paid.to_numpy())),
                                              list(to_cents(rl.line_amount.to_numpy())))
            realized.update(dict(zip(rl.invoice_line_id, res)))
        evidence = {"invoices": docs.invoice_id.tolist(), "invoice_lines": g.invoice_line_id.tolist(),
                    "po_lines": sorted(set(g.po_line_id.dropna())), "suppliers": [g.supplier_id.iloc[0]]}
        for r in g.itertuples():
            is_rep = r.dup_rank > 0
            amt_c = to_cents(r.line_amount)
            real_c = realized.get(r.invoice_line_id, 0) if is_rep else 0
            m = match.get(r.invoice_line_id)
            o_row = orig[orig.invoice_line_id == m].iloc[0] if m is not None else None
            same = bool(o_row is not None and o_row.quantity_invoiced == r.quantity_invoiced
                        and to_cents(o_row.unit_price) == to_cents(r.unit_price))
            if not is_rep:
                assess = "ORIGINAL_ENTRY"
            elif real_c > 0:
                assess = "CONFIRMED_DUPLICATE_PAYMENT"
            elif to_cents(r.amount_paid) > 0:
                assess = "REPEAT_PAID_ORIGINAL_NOT_FULLY_PAID"
            else:
                assess = "REPEAT_NOT_YET_PAID"
            details = {"role": "REPEAT" if is_rep else "ORIGINAL", "duplicate_rank": int(r.dup_rank),
                       "match_type": r.dup_match_type, "max_edit_distance_in_group": int(r.dup_edit_distance),
                       "confidence": "HIGH" if r.dup_match_type == "EXACT" else "MEDIUM",
                       "invoice_number": r.invoice_number, "normalized_invoice_number": r.invoice_number_norm,
                       "fuzzy_invoice_key": r.invoice_number_fuzzy,
                       "group_invoice_numbers": sorted({d["invoice_number"] for d in doc_info}),
                       "group_spans_same_po_lines": len({tuple(sorted(set(g.loc[g.invoice_id == x, "po_line_id"])))
                                                         for x in docs.invoice_id}) == 1,
                       "group_documents": doc_info, "original_invoice_id": orig_id,
                       "matched_original_line_id": m if is_rep else None,
                       "same_quantity_and_price_as_original": same if is_rep else None,
                       "line_amount": round(float(r.line_amount), 2), "amount_paid": round(float(r.amount_paid), 2),
                       "realized_duplicate_payment": from_cents(real_c), "payment_assessment": assess}
            if is_rep:
                how = ("" if r.dup_match_type != "FUZZY" else
                       f" (fuzzy match: numbers {' / '.join(sorted({d['invoice_number'] for d in doc_info}))} with an "
                       f"identical amount)")
                desc = (f"Supplier {r.supplier_id} invoice number '{r.invoice_number}' was already entered as "
                        f"{orig_id}{how}; {r.invoice_id} is repeat entry #{r.dup_rank}"
                        f"{' with the same quantity and price' if same else ''}. Potential duplicate payment "
                        f"{_usd(r.line_amount)}; realized duplicate payment so far {_usd(from_cents(real_c))} "
                        f"({assess.replace('_', ' ').lower()}).")
                basis = "line_amount of repeat entry (potential duplicate payment)"
            else:
                desc = (f"Original entry of supplier {r.supplier_id} invoice number '{r.invoice_number}', which "
                        f"was entered again as {', '.join(d['invoice_id'] for d in doc_info[1:])}. Exposure is "
                        f"counted on the repeat entry only.")
                basis = "0: original entry (exposure counted on the repeat entry)"
            rows.append({"invoice_line_id": r.invoice_line_id, "finding_type": DUP, "rule_id": RULE_IDS[DUP],
                         "potential_exposure": from_cents(amt_c if is_rep else 0),
                         "paid_exposure": from_cents(real_c), "confirmed_loss": from_cents(real_c),
                         "exposure_basis": basis, "evidence": evidence, "details": details,
                         "description": desc, "incident_key": f"DUP|{key}"})
    return pd.DataFrame(rows, columns=HIT_COLUMNS)


# ------------------------------------------------------------------------------------------------
# Rule B
# ------------------------------------------------------------------------------------------------
def rule_quantity_discrepancy(view: pd.DataFrame, config: DetectionConfig | None = None) -> pd.DataFrame:
    cfg = (config or DetectionConfig()).rules
    q = _quantity_status(view, cfg)
    cum_over = np.where(q.in_grace, view.cum_invoiced - view.approved_quantity,
                        view.cum_invoiced - view.supported_quantity)
    share = cum_over / view.cum_invoiced.where(view.cum_invoiced > 0)
    value = q.flag_units * view.approved_unit_price
    material = (share >= cfg.qty_materiality_pct) & (value >= cfg.qty_materiality_value)
    v = view[(q.flag_units > 0) & material].copy()
    if v.empty:
        return _empty()
    v["flag_units"] = q.flag_units[v.index]
    v["in_grace"] = q.in_grace[v.index]
    v["pending_units"] = q.pending_units[v.index]
    exp_c = (v.flag_units.to_numpy() * to_cents(v.approved_unit_price.to_numpy())).round().astype("int64")
    paid_c = prorate_paid(exp_c, to_cents(v.amount_paid.to_numpy()), to_cents(v.line_amount.to_numpy()))
    rows = []
    for r, e, p in zip(v.itertuples(), exp_c, paid_c):
        amd = [a for a in [r.applied_qty_amendment_id] if isinstance(a, str)]
        details = {"quantity_invoiced": float(r.quantity_invoiced), "previously_invoiced_on_po_line": float(r.previously_invoiced),
                   "cumulative_invoiced": float(r.cum_invoiced), "quantity_ordered": float(r.quantity_ordered),
                   "approved_quantity": float(r.approved_quantity), "applied_quantity_amendment_id": amd[0] if amd else None,
                   "received_total_all_receipts": float(r.received_total),
                   "received_as_of_invoice_date": float(r.received_as_of_invoice),
                   "receipts_dated_after_invoice": r.receipts_after_invoice, "receipt_ids": r.receipt_ids,
                   "receipt_dates": r.receipt_dates, "receipt_quantities": r.receipt_quantities,
                   "supported_quantity": float(r.supported_quantity), "excess_quantity": float(r.flag_units),
                   "excess_over_receipts": float(r.excess_qty), "excess_over_approved_quantity": float(r.excess_over_approved),
                   "invoice_age_days": int(r.invoice_age_days), "within_grace_period": bool(r.in_grace),
                   "pending_units_awaiting_receipt": float(r.pending_units),
                   "limit": "EXCEEDS_APPROVED_QUANTITY" if r.in_grace else r.qty_limit_reason,
                   "approved_unit_price": round(float(r.approved_unit_price), 2)}
        if r.in_grace:
            desc = (f"PO line {r.po_line_id}: cumulative invoiced {r.cum_invoiced:g} exceeds the approved quantity "
                    f"{r.approved_quantity:g} - no later receipt can justify it. This line adds {r.flag_units:g} unit(s) "
                    f"x approved price {_usd(r.approved_unit_price)} = {_usd(from_cents(e))}. (Invoice is "
                    f"{int(r.invoice_age_days)} days old, inside the {cfg.qty_grace_days}-day posting grace period; "
                    f"{r.pending_units:g} further unit(s) await receipt.)")
        else:
            limit = "total received" if r.qty_limit_reason == "EXCEEDS_RECEIPTS" else "approved order quantity"
            desc = (f"PO line {r.po_line_id}: cumulative invoiced {r.cum_invoiced:g} vs supported "
                    f"{r.supported_quantity:g} ({limit}; {len(r.receipt_ids)} receipt line(s), all dates included; "
                    f"invoice {int(r.invoice_age_days)} days old, past the {cfg.qty_grace_days}-day grace period). "
                    f"This line adds {r.flag_units:g} unsupported unit(s) x approved price "
                    f"{_usd(r.approved_unit_price)} = {_usd(from_cents(e))}.")
        rows.append({"invoice_line_id": r.invoice_line_id, "finding_type": QTY, "rule_id": RULE_IDS[QTY],
                     "potential_exposure": from_cents(e), "paid_exposure": from_cents(p), "confirmed_loss": 0.0,
                     "exposure_basis": "excess_quantity x approved_unit_price",
                     "evidence": {"invoice_lines": list(r.invoice_lines_on_po_line), "po_lines": [r.po_line_id],
                                  "receipts": list(r.receipt_ids), "amendments": amd, "invoices": [r.invoice_id]},
                     "details": details, "description": desc, "incident_key": None})
    return pd.DataFrame(rows, columns=HIT_COLUMNS)


def _quantity_status(view: pd.DataFrame, rules) -> pd.DataFrame:
    """Units to flag now vs units pending receipt, per invoice line."""
    in_grace = view.invoice_age_days < rules.qty_grace_days
    over_supported = view.cum_invoiced - view.supported_quantity
    over_approved = view.cum_invoiced - view.approved_quantity
    flag = np.where(in_grace, view.excess_over_approved, view.excess_qty)
    beyond_tol = np.where(in_grace, over_approved, over_supported) > rules.qty_tolerance_units
    flag = np.where(view.po_line_found & beyond_tol, flag, 0.0)
    pending = np.where(view.po_line_found & in_grace, view.excess_qty - view.excess_over_approved, 0.0)
    return pd.DataFrame({"flag_units": flag, "pending_units": pending, "in_grace": in_grace}, index=view.index)


PENDING_COLUMNS = ["invoice_line_id", "invoice_id", "supplier_id", "po_line_id", "invoice_date", "invoice_age_days",
                   "grace_days", "days_until_flag", "pending_units", "approved_unit_price",
                   "amount_at_stake_if_unreceived", "received_to_date", "cumulative_invoiced", "approved_quantity",
                   "status"]


def pending_quantity_checks(view: pd.DataFrame, config: DetectionConfig | None = None) -> pd.DataFrame:
    """Invoice lines billed beyond posted receipts but still inside the grace
    period. Not findings: they become Rule B findings if receipts are still
    missing once the invoice is older than ``qty_grace_days``."""
    cfg = (config or DetectionConfig()).rules
    q = _quantity_status(view, cfg)
    v = view[q.pending_units > 0]
    if v.empty:
        return pd.DataFrame(columns=PENDING_COLUMNS)
    pu = q.pending_units[v.index]
    return pd.DataFrame({
        "invoice_line_id": v.invoice_line_id, "invoice_id": v.invoice_id, "supplier_id": v.supplier_id,
        "po_line_id": v.po_line_id, "invoice_date": v.invoice_date.dt.strftime("%Y-%m-%d"),
        "invoice_age_days": v.invoice_age_days, "grace_days": cfg.qty_grace_days,
        "days_until_flag": cfg.qty_grace_days - v.invoice_age_days, "pending_units": pu,
        "approved_unit_price": v.approved_unit_price.round(2),
        "amount_at_stake_if_unreceived": from_cents((pu.to_numpy() * to_cents(v.approved_unit_price.to_numpy())).round().astype("int64")),
        "received_to_date": v.received_total, "cumulative_invoiced": v.cum_invoiced,
        "approved_quantity": v.approved_quantity, "status": "PENDING_RECEIPT"}).reset_index(drop=True)


# ------------------------------------------------------------------------------------------------
# Rule C
# ------------------------------------------------------------------------------------------------
def rule_po_price_mismatch(view: pd.DataFrame, config: DetectionConfig | None = None) -> pd.DataFrame:
    cfg = (config or DetectionConfig()).rules
    v = view[view.po_line_found & view.approved_unit_price.notna()]
    inv_c, appr_c = to_cents(v.unit_price.to_numpy()), to_cents(v.approved_unit_price.to_numpy())
    diff = inv_c - appr_c
    line_over = diff * v.quantity_invoiced.to_numpy().astype("int64")
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(appr_c > 0, diff / appr_c, np.inf)
    hit = (diff > 0) & (line_over > to_cents(cfg.price_materiality_abs)) & (pct > cfg.price_materiality_pct)
    v, inv_c, appr_c = v[hit], inv_c[hit], appr_c[hit]
    if v.empty:
        return _empty()
    exp_c = (inv_c - appr_c) * v.quantity_invoiced.to_numpy().astype("int64")
    paid_c = prorate_paid(exp_c, to_cents(v.amount_paid.to_numpy()), to_cents(v.line_amount.to_numpy()))
    rows = []
    for r, ic, ac, e, p in zip(v.itertuples(), inv_c, appr_c, exp_c, paid_c):
        unapproved = [{"amendment_id": a, "status": s, "requested_unit_price": None if pd.isna(pr) else round(float(pr), 2)}
                      for a, s, pr in r.unapproved_price_amendments]
        billed_unapproved = [u["amendment_id"] for u in unapproved
                             if u["requested_unit_price"] is not None and to_cents(u["requested_unit_price"]) == ic]
        applied = r.applied_price_amendment_id if isinstance(r.applied_price_amendment_id, str) else None
        pending_billed = [u["amendment_id"] for u in unapproved if u["status"] in cfg.pending_amendment_statuses
                          and u["amendment_id"] in billed_unapproved]
        borderline = False
        if pd.notna(r.next_approved_amendment_date):
            bdays = int(np.busday_count(r.invoice_date.date(), r.next_approved_amendment_date.date()))
            covers = ic <= to_cents(r.next_approved_amendment_price) + price_allowance_c(to_cents(r.next_approved_amendment_price), cfg)
            borderline = bool(covers and bdays <= cfg.amendment_borderline_business_days)
        if borderline:
            confidence, ambiguity = "LOW", "BORDERLINE_AMENDMENT_TIMING"
        elif pending_billed:
            confidence, ambiguity = "MEDIUM", "PENDING_AMENDMENT"
        else:
            confidence, ambiguity = "HIGH", None
        details = {"invoice_unit_price": from_cents(ic), "po_unit_price": round(float(r.po_unit_price), 2),
                   "approved_unit_price": from_cents(ac), "applied_price_amendment_id": applied,
                   "approved_amendments_dated_after_invoice": r.approved_price_amendments_after_invoice,
                   "unapproved_price_amendments": unapproved,
                   "billed_at_unapproved_amendment_price": billed_unapproved,
                   "price_difference_per_unit": from_cents(ic - ac),
                   "price_difference_pct": round((ic - ac) / ac * 100, 2) if ac else None,
                   "quantity_invoiced": float(r.quantity_invoiced),
                   "amount_paid": round(float(r.amount_paid), 2), "payment_status": r.payment_status,
                   "line_overcharge": from_cents(e), "materiality_rule": f"line overcharge > ${cfg.price_materiality_abs:.2f} "
                   f"AND variance > {cfg.price_materiality_pct:.1%}", "confidence": confidence, "ambiguity": ambiguity,
                   "next_approved_amendment_id": r.next_approved_amendment_id if pd.notna(r.next_approved_amendment_id) else None}
        note = ""
        if applied:
            note = f" after applying approved amendment {applied}"
        if billed_unapproved:
            note += f"; billed at the price requested in non-approved amendment {', '.join(billed_unapproved)}"
        if ambiguity == "BORDERLINE_AMENDMENT_TIMING":
            note += (f"; BORDERLINE: approved amendment {r.next_approved_amendment_id} covering this price is dated "
                     f"within {cfg.amendment_borderline_business_days} business days after the invoice")
        elif ambiguity == "PENDING_AMENDMENT":
            note += "; the requested price is in an amendment still pending approval (lower confidence)"
        desc = (f"Invoiced {_usd(from_cents(ic))}/unit vs approved {_usd(from_cents(ac))}/unit{note}. Potential "
                f"overcharge {_usd(from_cents(ic - ac))} x {r.quantity_invoiced:g} = {_usd(from_cents(e))} "
                f"(paid so far: {_usd(from_cents(p))}; not a confirmed loss).")
        rows.append({"invoice_line_id": r.invoice_line_id, "finding_type": PRICE, "rule_id": RULE_IDS[PRICE],
                     "potential_exposure": from_cents(e), "paid_exposure": from_cents(p), "confirmed_loss": 0.0,
                     "exposure_basis": "(invoice_unit_price - approved_unit_price) x quantity_invoiced",
                     "evidence": {"invoice_lines": [r.invoice_line_id], "invoices": [r.invoice_id],
                                  "po_lines": [r.po_line_id],
                                  "amendments": sorted(set(([applied] if applied else []) + list(r.all_amendment_ids)))},
                     "details": details, "description": desc, "incident_key": None})
    return pd.DataFrame(rows, columns=HIT_COLUMNS)


# ------------------------------------------------------------------------------------------------
# Rule D
# ------------------------------------------------------------------------------------------------
def rule_historical_price_spike(view: pd.DataFrame, history: pd.DataFrame,
                                config: DetectionConfig | None = None) -> pd.DataFrame:
    cfg = (config or DetectionConfig()).rules
    h = history.reindex(view.index)
    hit = h.hist_sufficient & (h.hist_ratio > h.hist_effective_threshold)
    v, h = view[hit], h[hit]
    if v.empty:
        return _empty()
    inv_c = to_cents(v.unit_price.to_numpy())
    known = v.po_line_found.to_numpy() & v.approved_unit_price.notna().to_numpy()
    appr_c = np.where(known, to_cents(v.approved_unit_price.fillna(0).to_numpy()), inv_c)
    over = inv_c - appr_c
    excess_c = np.where(over > price_allowance_c(appr_c, cfg), over, 0) * v.quantity_invoiced.to_numpy().astype("int64")
    paid_c = prorate_paid(excess_c, to_cents(v.amount_paid.to_numpy()), to_cents(v.line_amount.to_numpy()))
    rows = []
    for r, hr, ic, ac, e, p in zip(v.itertuples(), h.itertuples(), inv_c, appr_c, excess_c, paid_c):
        has_basis = bool(e > 0)
        details = {"invoice_unit_price": from_cents(ic), "baseline_statistic": cfg.hist_baseline,
                   "baseline_unit_price": round(float(hr.hist_baseline), 4), "n_prior_observations": int(hr.hist_n_prior),
                   "price_ratio": round(float(hr.hist_ratio), 4), "configured_ratio_threshold": cfg.hist_ratio_threshold,
                   "effective_ratio_threshold": round(float(hr.hist_effective_threshold), 4),
                   "robust_cv_of_history": None if pd.isna(hr.hist_robust_cv) else round(float(hr.hist_robust_cv), 4),
                   "prior_min": round(float(hr.hist_prior_min), 2), "prior_max": round(float(hr.hist_prior_max), 2),
                   "most_recent_prior_price": round(float(hr.hist_last_price), 2),
                   "prior_unit_prices": hr.hist_prior_prices, "prior_invoice_line_ids": hr.hist_prior_line_ids,
                   "approved_unit_price": round(float(r.approved_unit_price), 2),
                   "has_contractual_basis": has_basis,
                   "contractual_excess_per_unit": from_cents(int(max(0, ic - ac)) if has_basis else 0)}
        base = (f"Unit price {_usd(from_cents(ic))} is {hr.hist_ratio:.2f}x the {cfg.hist_baseline} "
                f"{_usd(hr.hist_baseline)} of {int(hr.hist_n_prior)} earlier purchases of {r.item_id} from "
                f"{r.supplier_id} (threshold {hr.hist_effective_threshold:.2f}x; earlier prices only).")
        if has_basis:
            desc = base + (f" The invoice also exceeds the approved PO price {_usd(r.approved_unit_price)}; "
                           f"contractual excess {_usd(from_cents(e))}.")
            basis = "contractual excess: (invoice_unit_price - approved_unit_price) x quantity_invoiced"
        else:
            desc = base + (" The invoice matches the approved PO price, so there is no contractual basis for a loss: "
                           "unusual price increase for review, exposure 0.")
            basis = "0: no contractual reference price is exceeded (unusual increase only)"
        rows.append({"invoice_line_id": r.invoice_line_id, "finding_type": HIST, "rule_id": RULE_IDS[HIST],
                     "potential_exposure": from_cents(e), "paid_exposure": from_cents(p), "confirmed_loss": 0.0,
                     "exposure_basis": basis,
                     "evidence": {"invoice_lines": [r.invoice_line_id] + list(hr.hist_prior_line_ids),
                                  "invoices": [r.invoice_id], "po_lines": [r.po_line_id]},
                     "details": details, "description": desc, "incident_key": None})
    return pd.DataFrame(rows, columns=HIT_COLUMNS)


def run_rules(view: pd.DataFrame, history: pd.DataFrame, config: DetectionConfig | None = None) -> pd.DataFrame:
    parts = [rule_duplicate_invoices(view, config), rule_quantity_discrepancy(view, config),
             rule_po_price_mismatch(view, config), rule_historical_price_spike(view, history, config)]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else _empty()
