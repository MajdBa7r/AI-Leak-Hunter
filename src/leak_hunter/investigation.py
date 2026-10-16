"""Investigation support: case assembly, calculations, playbooks, evidence map, reports.

A *case* is an incident from ``incidents.csv``. ``Workspace`` loads the saved
detection outputs plus the procurement tables (never the ground truth) and
answers, for one case:

  what was flagged and why          findings + plain-language description
  how the money was calculated      ``calculation_steps`` (formula with the actual numbers)
  which records support it          ``evidence_records`` (PO lines, receipts, amendments, invoices)
  what happened when                ``timeline``
  how the records connect           ``evidence_graph`` (nodes / edges for a layered evidence map)
  what to check next                ``playbook`` (verification steps specific to the evidence)
  a shareable summary               ``report_markdown``
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import ProcurementData, load_procurement_data

TYPE_LABEL = {"DUPLICATE_INVOICE": "Duplicate invoice", "QUANTITY_DISCREPANCY": "Quantity discrepancy",
              "PO_PRICE_MISMATCH": "PO price mismatch", "HISTORICAL_PRICE_SPIKE": "Historical price spike",
              "UNUSUAL_TRANSACTION": "Statistically unusual transaction"}
DISCLAIMER = ("Findings are indicators for human review, not proof of fraud. 'Potential exposure' is computed by "
              "documented formulas; only payments proven by records count as 'confirmed loss'.")


def usd(x) -> str:
    try:
        return f"${float(x):,.2f}"
    except (TypeError, ValueError):
        return "-"


def _j(s):
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(s) if isinstance(s, str) and s else {}
    except ValueError:
        return {}


@dataclass
class Workspace:
    out_dir: Path
    data: ProcurementData
    findings: pd.DataFrame
    incidents: pd.DataFrame
    recon: pd.DataFrame
    scores: pd.DataFrame | None
    pending: pd.DataFrame
    suppliers: pd.DataFrame
    summary: dict = field(default_factory=dict)

    @classmethod
    def load(cls, out_dir: str | Path = "outputs", data_dir: str | Path | None = None) -> "Workspace":
        out = Path(out_dir)
        need = ["findings.csv", "incidents.csv", "line_reconciliation.csv", "run_summary.json"]
        miss = [n for n in need if not (out / n).is_file()]
        if miss:
            raise FileNotFoundError(f"Missing detection outputs {miss} in {out}. Run: python -m leak_hunter.pipeline "
                                    f"--out-dir {out}")
        summary = json.loads((out / "run_summary.json").read_text())
        # portable: the data folder recorded in run_summary.json is used only if it exists here
        # (outputs produced on another machine fall back to the repository's own dataset)
        recorded = summary.get("data_dir")
        if data_dir is None and recorded and not Path(recorded).is_dir():
            recorded = None
        data = load_procurement_data(data_dir or recorded)
        rd = lambda n, **k: pd.read_csv(out / n, **k) if (out / n).is_file() else pd.DataFrame()
        return cls(out, data,
                   rd("findings.csv", dtype={"invoice_number": str}),
                   rd("incidents.csv", keep_default_na=False, na_values={c: [""] for c in
                      ["max_anomaly_score", "rule_confidence", "if_percentile"]}),
                   rd("line_reconciliation.csv", parse_dates=["invoice_date"]),
                   rd("if_scores.csv") if (out / "if_scores.csv").is_file() else None,
                   rd("pending_quantity_checks.csv"), rd("supplier_profiles.csv"), summary)

    # ------------------------------------------------------------------ case assembly
    def incident(self, incident_id: str) -> pd.Series:
        m = self.incidents[self.incidents.incident_id == incident_id]
        if m.empty:
            raise KeyError(f"Unknown incident {incident_id}")
        return m.iloc[0]

    def case_findings(self, incident_id: str) -> pd.DataFrame:
        f = self.findings[self.findings.incident_id == incident_id].copy()
        f["details_obj"] = f.details.map(_j)
        f["evidence_obj"] = f.evidence_record_ids.map(_j)
        order = {"DUPLICATE_INVOICE": 0, "QUANTITY_DISCREPANCY": 1, "PO_PRICE_MISMATCH": 2,
                 "HISTORICAL_PRICE_SPIKE": 3, "UNUSUAL_TRANSACTION": 4}
        return f.sort_values(by="finding_type", key=lambda s: s.map(order)).reset_index(drop=True)

    def evidence_ids(self, incident_id: str) -> dict[str, list[str]]:
        ev: dict[str, set] = {}
        for e in self.case_findings(incident_id).evidence_obj:
            for k, v in e.items():
                ev.setdefault(k, set()).update(v)
        return {k: sorted(v) for k, v in ev.items()}

    def evidence_records(self, incident_id: str) -> dict[str, pd.DataFrame]:
        """Source records behind a case. Historical-baseline lines are listed separately."""
        f = self.case_findings(incident_id)
        ev = self.evidence_ids(incident_id)
        case_lines = set(self.incident(incident_id).invoice_line_ids.split("|"))
        hist = {l for d in f.details_obj for l in d.get("prior_invoice_line_ids", [])}
        pol = set(ev.get("po_lines", []))
        inv = self.data.invoices
        lines_ev = (set(ev.get("invoice_lines", [])) - hist) | case_lines
        rec = self.recon.set_index("invoice_line_id")
        il = inv[inv.invoice_line_id.isin(lines_ev) | inv.po_line_id.isin(pol)].copy()
        il["in_case"] = il.invoice_line_id.isin(case_lines)
        il["approved_unit_price"] = il.invoice_line_id.map(rec.approved_unit_price)
        po = self.data.purchase_orders[self.data.purchase_orders.po_line_id.isin(pol)]
        gr = self.data.goods_receipts[self.data.goods_receipts.po_line_id.isin(pol)]
        am = self.data.po_amendments[self.data.po_amendments.po_line_id.isin(pol)]
        hl = inv[inv.invoice_line_id.isin(hist)]
        sup_ids = set(f.supplier_id)
        sup = self.suppliers[self.suppliers.supplier_id.isin(sup_ids)] if len(self.suppliers) else pd.DataFrame()
        return {"invoice_lines": il.sort_values(["invoice_date", "invoice_line_id"]),
                "purchase_order_lines": po, "goods_receipts": gr.sort_values("receipt_date"),
                "amendments": am, "price_history_lines": hl.sort_values("invoice_date"), "supplier": sup}

    # ------------------------------------------------------------------ calculations
    @staticmethod
    def calculation_steps(finding: pd.Series) -> list[str]:
        d, t = _j(finding.details), finding.finding_type
        if t == "DUPLICATE_INVOICE":
            docs = d.get("group_documents", [])
            s = [f"Match type: {d.get('match_type')} on invoice numbers {', '.join(d.get('group_invoice_numbers', []))} "
                 f"(supplier {finding.supplier_id})."]
            s += [f"{x['role'].title()} {x['invoice_id']} dated {x['invoice_date']}: amount {usd(x['document_amount'])}, "
                  f"paid {usd(x['amount_paid'])} ({x['payment_status']})." for x in docs]
            if d.get("role") == "REPEAT":
                s += [f"Potential exposure = amount of the repeat line = {usd(finding.potential_exposure)}.",
                      f"Realized duplicate payment = max(0, paid on all entries - original amount) = "
                      f"{usd(d.get('realized_duplicate_payment'))} -> {d.get('payment_assessment', '').replace('_', ' ').lower()}."]
            else:
                s.append("Original entry: exposure 0 here, counted once on the repeat entry.")
            return s
        if t == "QUANTITY_DISCREPANCY":
            rc = ", ".join(f"{i} ({q:g} on {dt})" for i, q, dt in zip(d.get("receipt_ids", []), d.get("receipt_quantities", []),
                                                                         d.get("receipt_dates", [])))
            return [f"Approved quantity {d.get('approved_quantity'):g} (ordered {d.get('quantity_ordered'):g}"
                    + (f", amended by {d['applied_quantity_amendment_id']})" if d.get("applied_quantity_amendment_id") else ")")
                    + f"; received in total {d.get('received_total_all_receipts'):g}: {rc or 'no receipts'}.",
                    f"Supported quantity = min(received, approved) = {d.get('supported_quantity'):g}.",
                    f"Invoiced on this PO line so far = {d.get('cumulative_invoiced'):g} "
                    f"(this line {d.get('quantity_invoiced'):g}, earlier {d.get('previously_invoiced_on_po_line'):g}).",
                    f"Unsupported units attributed to this line = {d.get('excess_quantity'):g}"
                    + (" (inside grace period: only units beyond the approved quantity)" if d.get("within_grace_period") else "")
                    + ".",
                    f"Potential exposure = {d.get('excess_quantity'):g} x approved price {usd(d.get('approved_unit_price'))} "
                    f"= {usd(finding.potential_exposure)}; already paid share {usd(finding.paid_exposure)}."]
        if t == "PO_PRICE_MISMATCH":
            s = [f"PO price {usd(d.get('po_unit_price'))}"
                 + (f", amended to {usd(d.get('approved_unit_price'))} by {d['applied_price_amendment_id']}"
                    if d.get("applied_price_amendment_id") else "") + f" -> approved {usd(d.get('approved_unit_price'))}/unit.",
                 f"Invoiced {usd(d.get('invoice_unit_price'))}/unit: +{usd(d.get('price_difference_per_unit'))} "
                 f"({d.get('price_difference_pct')}%), above materiality ({d.get('materiality_rule', 'policy')}).",
                 f"Potential exposure = {usd(d.get('price_difference_per_unit'))} x {d.get('quantity_invoiced'):g} units = "
                 f"{usd(finding.potential_exposure)}; already paid share {usd(finding.paid_exposure)}."]
            if d.get("billed_at_unapproved_amendment_price"):
                s.append(f"Billed exactly at the price requested in non-approved amendment(s) "
                         f"{', '.join(d['billed_at_unapproved_amendment_price'])}.")
            if d.get("ambiguity"):
                s.append(f"Confidence {d.get('confidence')}: {d['ambiguity'].replace('_', ' ').lower()}.")
            return s
        if t == "HISTORICAL_PRICE_SPIKE":
            return [f"Earlier prices for this item from this supplier ({d.get('n_prior_observations')} purchases): "
                    f"{usd(d.get('prior_min'))} - {usd(d.get('prior_max'))}, median {usd(d.get('baseline_unit_price'))}.",
                    f"Invoice price {usd(d.get('invoice_unit_price'))} = {d.get('price_ratio')}x the median "
                    f"(threshold {d.get('effective_ratio_threshold')}x, earlier invoices only).",
                    (f"Approved PO price {usd(d.get('approved_unit_price'))} is also exceeded: contractual exposure "
                     f"{usd(finding.potential_exposure)}." if d.get("has_contractual_basis") else
                     f"The approved PO price {usd(d.get('approved_unit_price'))} matches the invoice: no contractual "
                     f"basis, exposure 0 - question the PO price itself.")]
        if t == "UNUSUAL_TRANSACTION":
            tf = d.get("top_features", [])
            return ([f"Isolation Forest score {d.get('anomaly_score', 0):.3f} (rank {d.get('risk_rank')}, threshold "
                     f"{d.get('review_threshold', 0):.3f}); not a probability."] +
                    [f"{x['feature']} = {x['value']} (robust z {x['robust_z']})" for x in tf] +
                    ["No monetary exposure is attached to statistical anomalies."])
        return []

    # ------------------------------------------------------------------ playbook
    def playbook(self, incident_id: str) -> list[dict]:
        """Next verification steps, specific to the evidence. Each: step, why, records."""
        steps: list[dict] = []
        inc = self.incident(incident_id)
        cf = self.case_findings(incident_id)
        dup_repeat_lines = set(cf[(cf.finding_type == "DUPLICATE_INVOICE")
                                  & cf.details_obj.map(lambda d: d.get("role") == "REPEAT")].invoice_line_id)
        for _, f in cf.iterrows():
            d, t = f.details_obj, f.finding_type
            if t == "QUANTITY_DISCREPANCY" and f.invoice_line_id in dup_repeat_lines:
                continue   # the excess IS the duplicated quantity: resolved by the duplicate steps, counted once
            if t == "DUPLICATE_INVOICE" and d.get("role") == "REPEAT":
                docs = d.get("group_documents", [])
                ids = [x["invoice_id"] for x in docs]
                if f.amount_paid == 0:
                    steps.append(dict(step=f"Place a payment block on {f.invoice_id} before the next payment run",
                                      why="Repeat entry not yet paid - the exposure is still preventable", records=[f.invoice_id]))
                steps.append(dict(step=f"Match bank remittances for {', '.join(ids)}",
                                  why=f"Payment records show realized duplicate payment of {usd(d.get('realized_duplicate_payment'))}",
                                  records=ids))
                steps.append(dict(step=f"Request the supplier statement of account for invoice number "
                                       f"{d.get('invoice_number')}", why="Confirms whether the supplier billed once or twice",
                                  records=[f.supplier_id]))
                if d.get("match_type") == "FUZZY":
                    steps.append(dict(step="Compare the two scanned invoice documents side by side",
                                      why=f"Numbers differ slightly ({', '.join(d.get('group_invoice_numbers', []))}); "
                                          "a re-keyed copy is likely but must be confirmed", records=ids))
                if d.get("realized_duplicate_payment", 0) > 0:
                    steps.append(dict(step=f"Raise a recovery claim / credit note for {usd(d['realized_duplicate_payment'])}",
                                      why="Duplicate payment proven by payment records", records=ids))
            elif t == "QUANTITY_DISCREPANCY":
                steps.append(dict(step=f"Confirm physical receipt with the warehouse for PO line {f.po_line_id}",
                                  why=f"Posted receipts support {d.get('supported_quantity'):g} units; "
                                      f"{d.get('cumulative_invoiced'):g} were invoiced",
                                  records=[f.po_line_id] + d.get("receipt_ids", [])))
                steps.append(dict(step="Look for unposted goods receipts, delivery notes or returns on this PO line",
                                  why="Posting lag and returns are the most common legitimate explanations",
                                  records=[f.po_line_id]))
                steps.append(dict(step=f"If the shortfall is confirmed, request a credit note for {usd(f.potential_exposure)}"
                                       + (" and block the unpaid balance" if f.amount_paid < f.line_amount else ""),
                                  why=f"{d.get('excess_quantity'):g} unsupported units x {usd(d.get('approved_unit_price'))}",
                                  records=[f.invoice_id]))
            elif t == "PO_PRICE_MISMATCH":
                amd = [a for a in [d.get("applied_price_amendment_id")] if a]
                steps.append(dict(step=f"Check the contract / price list for {f.item_id}: approved price "
                                       f"{usd(d.get('approved_unit_price'))}",
                                  why=f"Invoiced {usd(d.get('invoice_unit_price'))} (+{d.get('price_difference_pct')}%)",
                                  records=[f.po_line_id] + amd))
                if d.get("billed_at_unapproved_amendment_price"):
                    steps.append(dict(step="Confirm the supplier was told the price amendment was not approved",
                                      why="Invoice uses the price from a non-approved amendment",
                                      records=d["billed_at_unapproved_amendment_price"]))
                if d.get("ambiguity") == "BORDERLINE_AMENDMENT_TIMING":
                    steps.append(dict(step=f"Ask the category manager for the effective date of {d.get('next_approved_amendment_id')}",
                                      why="It was approved within a few business days after the invoice - may be retroactive",
                                      records=[d.get("next_approved_amendment_id")]))
                steps.append(dict(step=f"If unauthorized, request a credit note for {usd(f.potential_exposure)}",
                                  why="(invoice - approved) x quantity", records=[f.invoice_id]))
            elif t == "HISTORICAL_PRICE_SPIKE":
                steps.append(dict(step=f"Benchmark the price of {f.item_id} against the market / other suppliers",
                                  why=f"{d.get('price_ratio')}x this supplier's earlier median {usd(d.get('baseline_unit_price'))}",
                                  records=d.get("prior_invoice_line_ids", [])[-5:]))
                if not d.get("has_contractual_basis"):
                    steps.append(dict(step=f"Ask the PO approver why PO line {f.po_line_id} was raised at this price",
                                      why="The PO itself carries the high price - the control gap is upstream of invoicing",
                                      records=[f.po_line_id]))
            elif t == "UNUSUAL_TRANSACTION":
                tf = ", ".join(x["feature"] for x in f.details_obj.get("top_features", []))
                steps.append(dict(step="Review the unusual characteristics against the supplier's recent invoices",
                                  why=f"Main statistical drivers: {tf}", records=[f.invoice_line_id]))
                if inc.evidence_tier == "UNUSUAL_PATTERN":
                    steps.append(dict(step="Close as NO_ISSUE with a note if a business reason explains it",
                                      why="No rule was violated; the model only flags rarity", records=[]))
        seen, out = set(), []
        for s in steps:
            if s["step"] not in seen:
                seen.add(s["step"]); out.append(s)
        return out

    def exposure_breakdown(self, incident_id: str) -> pd.DataFrame:
        """Per line: rule components and the de-duplicated amount that counts.
        line exposure = max(duplicate, quantity + max(price mismatch, price spike))."""
        from .exposure import line_exposure
        f = self.findings[self.findings.incident_id == incident_id]
        rb = f[f.detection_method == "RULE_BASED"]
        if rb.empty:
            return pd.DataFrame(columns=["invoice_line_id", "duplicate", "quantity", "price_mismatch", "price_spike",
                                         "counted_exposure", "counted_paid", "confirmed_loss"])
        piv = rb.pivot_table(index="invoice_line_id", columns="finding_type", values="potential_exposure",
                             aggfunc="max", fill_value=0.0)
        le = line_exposure(f).set_index("invoice_line_id")
        out = pd.DataFrame({
            "duplicate": piv.get("DUPLICATE_INVOICE", 0.0), "quantity": piv.get("QUANTITY_DISCREPANCY", 0.0),
            "price_mismatch": piv.get("PO_PRICE_MISMATCH", 0.0), "price_spike": piv.get("HISTORICAL_PRICE_SPIKE", 0.0)},
            index=piv.index)
        out["counted_exposure"] = le.line_exposure
        out["counted_paid"] = le.line_paid_exposure
        out["confirmed_loss"] = le.line_confirmed_loss
        return out.reset_index().round(2)

    # ------------------------------------------------------------------ timeline and graph
    def timeline(self, incident_id: str) -> pd.DataFrame:
        r = self.evidence_records(incident_id)
        ev = []
        for x in r["purchase_order_lines"].itertuples():
            ev.append((x.po_date, "PO line", x.po_line_id, f"{x.quantity_ordered:g} x {usd(x.unit_price)}"))
        for x in r["amendments"].itertuples():
            what = (f"price {usd(x.previous_unit_price)} -> {usd(x.new_unit_price)}" if x.amendment_type == "PRICE"
                    else f"qty {x.previous_quantity} -> {x.new_quantity}")
            ev.append((x.amendment_date, f"Amendment ({x.status})", x.amendment_id, what))
        for x in r["goods_receipts"].itertuples():
            ev.append((x.receipt_date, "Goods receipt", x.receipt_id, f"{x.quantity_received:g} units on {x.po_line_id}"))
        for x in r["invoice_lines"].itertuples():
            ev.append((x.invoice_date, "Invoice (flagged)" if x.in_case else "Invoice", x.invoice_line_id,
                       f"#{x.invoice_number}: {x.quantity_invoiced:g} x {usd(x.unit_price)}; {x.payment_status}"))
        t = pd.DataFrame(ev, columns=["date", "event", "record_id", "detail"])
        return t.sort_values(["date", "event"]).reset_index(drop=True)

    def evidence_graph(self, incident_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Layered evidence map: Supplier | PO lines | Amendments & receipts | Invoice lines | Invoices."""
        r = self.evidence_records(incident_id)
        f = self.case_findings(incident_id)
        flagged = set(f.invoice_line_id)
        nodes, edges = [], []

        def add(nid, kind, layer, label, hot=False):
            nodes.append(dict(id=nid, kind=kind, layer=layer, label=label, flagged=hot))

        for s in sorted(set(f.supplier_id)):
            add(s, "Supplier", 0, s)
        for x in r["purchase_order_lines"].itertuples():
            add(x.po_line_id, "PO line", 1, f"{x.po_line_id}\n{x.quantity_ordered:g} x {usd(x.unit_price)}")
            edges.append((x.supplier_id, x.po_line_id, "ordered"))
        for x in r["amendments"].itertuples():
            add(x.amendment_id, f"Amendment ({x.status})", 2, f"{x.amendment_id}\n{x.amendment_type} {x.status}")
            edges.append((x.po_line_id, x.amendment_id, "amended"))
        for x in r["goods_receipts"].itertuples():
            add(x.receipt_id, "Goods receipt", 2, f"{x.receipt_id}\n{x.quantity_received:g} units")
            edges.append((x.po_line_id, x.receipt_id, "received"))
        for x in r["invoice_lines"].itertuples():
            add(x.invoice_line_id, "Invoice line", 3, f"{x.invoice_line_id}\n{x.quantity_invoiced:g} x {usd(x.unit_price)}",
                x.invoice_line_id in flagged)
            edges.append((x.po_line_id, x.invoice_line_id, "billed"))
        docs = r["invoice_lines"].drop_duplicates("invoice_id")
        for x in docs.itertuples():
            add(x.invoice_id, "Invoice", 4, f"{x.invoice_id}\n#{x.invoice_number}\n{x.payment_status}",
                x.invoice_id in set(f.invoice_id))
        for x in r["invoice_lines"].itertuples():
            edges.append((x.invoice_line_id, x.invoice_id, "part of"))
        dup = f[f.finding_type == "DUPLICATE_INVOICE"]
        if len(dup):
            ids = sorted(set(dup.invoice_id))
            for a, b in zip(ids, ids[1:]):
                edges.append((a, b, "same invoice number"))
        n = pd.DataFrame(nodes).drop_duplicates("id")
        e = pd.DataFrame(edges, columns=["source", "target", "relation"]).drop_duplicates()
        e = e[e.source.isin(n.id) & e.target.isin(n.id)]
        n["y"] = n.groupby("layer").cumcount()
        n["y"] = n.y - n.groupby("layer").y.transform("max") / 2
        return n.reset_index(drop=True), e.reset_index(drop=True)

    # ------------------------------------------------------------------ report
    def report_markdown(self, incident_id: str, decisions: pd.DataFrame | None = None) -> str:
        inc = self.incident(incident_id)
        f = self.case_findings(incident_id)
        r = self.evidence_records(incident_id)
        L = [f"# Case {incident_id}", "",
             f"**Priority:** rank {int(inc.priority_rank)} of {len(self.incidents)} - band {inc.band} ({inc.band_label}), "
             f"risk score {inc.risk_score:.1f}/100  ",
             f"**Supplier:** {inc.supplier_id}  ",
             f"**Evidence tier:** {inc.evidence_tier.replace('_', ' ').title()}  ",
             f"**Potential exposure:** {usd(inc.potential_exposure)} | **already paid:** {usd(inc.paid_exposure)} | "
             f"**confirmed loss:** {usd(inc.confirmed_loss)}", "",
             "## Risk score breakdown", "",
             "| Component | Points |", "|---|---|",
             f"| Exposure | {inc.pts_exposure:.1f} |", f"| Evidence ({inc.rule_confidence if isinstance(inc.rule_confidence, str) else 'none'}) | {inc.pts_evidence:.1f} |",
             f"| Money already paid | {inc.pts_realization:.1f} |", f"| Statistical anomaly (Isolation Forest) | {inc.pts_anomaly:.1f} |",
             f"| **Total** | **{inc.risk_score:.1f}** |", "",
             "## How the exposure is counted", "",
             "Each line counts max(duplicate, quantity + max(price mismatch, price spike)) - overlapping rules never "
             "add up twice.", "", _md_table(self.exposure_breakdown(incident_id)), "", "## Findings", ""]
        for _, x in f.iterrows():
            L += [f"### {TYPE_LABEL.get(x.finding_type, x.finding_type)} - {x.invoice_line_id} ({x.detection_method}, {x.severity})",
                  "", x.description, "", "Calculation:", ""] + [f"1. {s}" for s in self.calculation_steps(x)] + [""]
        L += ["## Recommended next steps", ""]
        L += [f"- [ ] **{s['step']}** - {s['why']}" + (f" (records: {', '.join(map(str, s['records']))})" if s["records"] else "")
              for s in self.playbook(incident_id)]
        L += ["", "## Supporting records", ""]
        for name, cols in [("invoice_lines", ["invoice_line_id", "invoice_id", "invoice_number", "invoice_date", "quantity_invoiced",
                                              "unit_price", "approved_unit_price", "payment_status", "amount_paid", "in_case"]),
                           ("purchase_order_lines", ["po_line_id", "po_date", "item_id", "quantity_ordered", "unit_price",
                                                     "approved_amendment_id"]),
                           ("goods_receipts", ["receipt_id", "po_line_id", "receipt_date", "quantity_received"]),
                           ("amendments", ["amendment_id", "po_line_id", "amendment_date", "amendment_type", "status",
                                           "previous_unit_price", "new_unit_price", "previous_quantity", "new_quantity"])]:
            df = r[name]
            if len(df):
                L += [f"**{name.replace('_', ' ').title()}**", "", _md_table(df[[c for c in cols if c in df.columns]]), ""]
        if decisions is not None and len(decisions):
            L += ["## Decision log", "", _md_table(decisions[["created_at", "reviewer", "status", "recovered_amount", "note"]]), ""]
        L += ["---", f"_{DISCLAIMER}_"]
        return "\n".join(L)


def _md_table(df: pd.DataFrame) -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%d")
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)
