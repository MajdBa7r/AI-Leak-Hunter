"""AI Procurement Investigation Copilot - grounded on structured case evidence.

Design
------
* One grounded payload per question. ``case_context`` (Case File) and
  ``portfolio_context`` (Overview) build a JSON document from the detection outputs:
  findings, calculation steps, de-duplicated exposure, risk-score breakdown,
  supplier profile, linked cases, timeline, source records, playbook and the rule
  thresholds actually used. Nothing else is given to the model.
* Providers (optional): OpenAI, Groq (OpenAI-compatible API) or Google Gemini,
  called over HTTPS with ``requests``. Selected from environment variables or
  Streamlit secrets: OPENAI_API_KEY / GROQ_API_KEY / GEMINI_API_KEY (or
  GOOGLE_API_KEY), optional LEAK_HUNTER_LLM_PROVIDER and *_MODEL overrides.
* Offline copilot: with no key, a forced offline switch, or ANY provider error
  (timeout, network, quota, bad model), answers come from deterministic
  templates over the same payload, so the chat never fails.
* Citation check: every transaction ID and dollar amount in a model answer is
  looked up in the payload; anything not found is reported under the answer.

The copilot never decides that something is fraud; it reports indicators,
calculations and the records behind them.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

import pandas as pd

from .investigation import TYPE_LABEL, Workspace, usd

QUICK_PROMPTS = {
    "summary": "🔍 Summarize the financial risk and calculation",
    "evidence": "⚠️ What evidence proves this is an overcharge / duplicate?",
    "letter": "📋 Draft a formal inquiry letter to this supplier",
}
OVERVIEW_PROMPTS = {
    "top": "🔍 Which cases should we investigate first, and why?",
    "money": "💰 How much money is at risk, and how much is already paid?",
    "suppliers": "🏭 Which suppliers concentrate the most exposure?",
}
PROVIDERS = {
    "openai": {"key": ["OPENAI_API_KEY"], "model_env": "OPENAI_MODEL", "default_model": "gpt-4o-mini",
               "url": "https://api.openai.com/v1/chat/completions"},
    "groq": {"key": ["GROQ_API_KEY"], "model_env": "GROQ_MODEL", "default_model": "llama-3.3-70b-versatile",
             "url": "https://api.groq.com/openai/v1/chat/completions"},
    "gemini": {"key": ["GEMINI_API_KEY", "GOOGLE_API_KEY"], "model_env": "GEMINI_MODEL",
               "default_model": "gemini-2.5-flash",
               "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"},
}
SYSTEM_PROMPT = """You are the AI Leak Hunter investigation copilot: a meticulous procurement auditor.
Answer ONLY from the JSON evidence in <evidence>. It was produced by deterministic reconciliation rules and an
anomaly model; treat it as the complete record available to you.

Rules:
1. Cite transaction IDs (invoice, invoice line, PO line, receipt, amendment, supplier, case) and dollar amounts
   exactly as they appear in the evidence. Prefer short quotes of the calculation steps.
2. Keep the distinctions: "potential exposure" (formula estimate), "already paid" (part of it paid out),
   "confirmed loss" (only duplicate payments proven by payment records). Never call anything fraud or theft;
   say "indicator", "discrepancy" or "requires verification".
3. Quote the thresholds that applied (materiality, grace period, ratio thresholds) when explaining a flag.
4. If the evidence does not contain the answer, say exactly: "The evidence available to me does not show that."
   and name the record or check that would resolve it. Do not guess, do not use outside knowledge about
   real companies, prices or laws.
5. Letters to suppliers: formal, neutral, non-accusatory; request documents or a credit note; list the exact
   invoice numbers, IDs and amounts from the evidence; use placeholders like [Your name] and [Organization]
   for anything not in the evidence.
6. Be concise: short paragraphs or bullet points, markdown allowed, no tables wider than 5 columns."""

ID_RE = re.compile(r"\b(?:INC-[A-Z0-9-]+|INV-\d{6}(?:-\d{2})?|PO-\d{6}(?:-\d{2})?|GR-\d{6}(?:-RET)?|AMD-\d{5}"
                   r"|SUP-\d{4}(?:-B)?|ITM-\d{4})\b")
AMOUNT_RE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")


# ------------------------------------------------------------------ grounded payloads
def _df_records(df: pd.DataFrame, cols: list[str], limit: int) -> list[dict]:
    if df is None or df.empty:
        return []
    d = df[[c for c in cols if c in df.columns]].head(limit).copy()
    for c in d.columns:
        if pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = d[c].dt.strftime("%Y-%m-%d")
    return json.loads(d.to_json(orient="records", date_format="iso"))


def _compact_details(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, list) and len(v) > 12:
            out[k] = v[:6] + [f"... {len(v) - 12} more ..."] + v[-6:]
        else:
            out[k] = v
    return out


def thresholds(ws: Workspace) -> dict:
    r = (ws.summary.get("config") or {}).get("rules", {})
    return {
        "price_mismatch_materiality": f"line overcharge > ${r.get('price_materiality_abs', 5):.2f} AND unit-price "
                                      f"variance > {100 * r.get('price_materiality_pct', 0.02):.1f}% of approved price",
        "quantity_materiality": f"excess >= {100 * r.get('qty_materiality_pct', 0.015):.1f}% of invoiced quantity AND "
                                f">= ${r.get('qty_materiality_value', 25):.2f}",
        "posting_lag_grace_period_days": r.get("qty_grace_days", 30),
        "historical_spike": f"price > {r.get('hist_ratio_threshold', 1.3)}x median of >= {r.get('hist_min_prior', 5)} "
                            "strictly earlier purchases (threshold widens for volatile price histories)",
        "duplicate_matching": f"same supplier; exact number, or fuzzy (edit distance <= {r.get('dup_fuzzy_max_distance', 2)}"
                              f", identical amount, within {r.get('dup_fuzzy_window_days', 60)} days, same items)",
        "exposure_overlap_rule": "per line: max(duplicate, quantity + max(price mismatch, price spike))",
    }


def case_context(ws: Workspace, incident_id: str) -> dict:
    inc = ws.incident(incident_id)
    f = ws.case_findings(incident_id)
    rec = ws.evidence_records(incident_id)
    sup = ws.suppliers[ws.suppliers.supplier_id == inc.supplier_id]
    linked = [x for x in str(inc.get("related_incident_ids") or "").split("|") if x and x != "nan"]
    lk = ws.incidents[ws.incidents.incident_id.isin(linked)]
    return {
        "scope": "case",
        "case": {
            "case_id": incident_id, "queue_rank": int(inc.priority_rank), "cases_in_queue": len(ws.incidents),
            "band": f"{inc.band} ({inc.band_label})", "case_type": TYPE_LABEL.get(inc.incident_type, inc.incident_type),
            "finding_types": [TYPE_LABEL.get(t, t) for t in str(inc.finding_types).split("|")],
            "detected_by": str(inc.detection_methods).replace("RULE_BASED", "business rules")
                           .replace("ISOLATION_FOREST", "anomaly model"),
            "supplier_id": inc.supplier_id, "invoice_ids": str(inc.invoice_ids).split("|"),
            "invoice_line_ids": str(inc.invoice_line_ids).split("|"),
            "evidence_tier": str(inc.evidence_tier).replace("_", " ").lower(),
            "rule_confidence": inc.rule_confidence if isinstance(inc.rule_confidence, str) else "n/a",
            "potential_exposure_usd": round(float(inc.potential_exposure), 2),
            "already_paid_usd": round(float(inc.paid_exposure), 2),
            "confirmed_loss_usd": round(float(inc.confirmed_loss), 2),
            "expected_value_usd": round(float(inc.get("expected_value", 0) or 0), 2),
            "risk_score": {"total": float(inc.risk_score), "exposure_points_of_35": float(inc.pts_exposure),
                           "evidence_points_of_30": float(inc.pts_evidence),
                           "money_paid_points_of_15": float(inc.pts_realization),
                           "anomaly_model_points_of_20": float(inc.pts_anomaly),
                           "anomaly_model_percentile": round(float(inc.if_percentile), 4)
                           if pd.notna(inc.if_percentile) else None},
        },
        "findings": [{"finding_id": x.finding_id, "type": TYPE_LABEL.get(x.finding_type, x.finding_type),
                      "method": "business rule" if x.detection_method == "RULE_BASED" else "anomaly model",
                      "invoice_line_id": x.invoice_line_id, "invoice_id": x.invoice_id,
                      "supplier_invoice_number": x.invoice_number, "po_line_id": x.po_line_id, "item_id": x.item_id,
                      "invoice_date": x.invoice_date, "severity": x.severity,
                      "potential_exposure_usd": x.potential_exposure, "already_paid_usd": x.paid_exposure,
                      "confirmed_loss_usd": x.confirmed_loss, "line_amount_usd": x.line_amount,
                      "amount_paid_usd": x.amount_paid, "exposure_formula": x.exposure_basis,
                      "description": x.description, "calculation_steps": ws.calculation_steps(x),
                      "details": _compact_details(x.details_obj)} for x in f.itertuples()],
        "exposure_breakdown_per_line": _df_records(ws.exposure_breakdown(incident_id),
                                                   list(ws.exposure_breakdown(incident_id).columns), 20),
        "thresholds_applied": thresholds(ws),
        "supplier_profile": _df_records(sup, ["supplier_id", "supplier_name", "category", "country", "supplier_tier",
                                              "invoices", "invoice_lines", "spend", "incidents", "p1_incidents",
                                              "potential_exposure", "confirmed_loss", "exposure_share_of_spend",
                                              "months_with_incidents", "incident_types"], 1),
        "linked_cases": _df_records(lk, ["incident_id", "incident_type", "potential_exposure", "band", "risk_score"], 10),
        "timeline": _df_records(ws.timeline(incident_id), ["date", "event", "record_id", "detail"], 30),
        "records": {
            "invoice_lines": _df_records(rec["invoice_lines"], ["invoice_line_id", "invoice_id", "invoice_number",
                                         "invoice_date", "po_line_id", "quantity_invoiced", "unit_price",
                                         "approved_unit_price", "line_amount", "payment_status", "amount_paid",
                                         "in_case"], 15),
            "purchase_order_lines": _df_records(rec["purchase_order_lines"], ["po_line_id", "po_date", "item_id",
                                                "item_description", "quantity_ordered", "unit_price",
                                                "approved_amendment_id"], 10),
            "goods_receipts": _df_records(rec["goods_receipts"], ["receipt_id", "po_line_id", "receipt_date",
                                                                  "quantity_received"], 15),
            "amendments": _df_records(rec["amendments"], ["amendment_id", "po_line_id", "amendment_date",
                                                          "amendment_type", "status", "previous_unit_price",
                                                          "new_unit_price", "previous_quantity", "new_quantity"], 10),
        },
        "recommended_next_steps": ws.playbook(incident_id),
        "disclaimer": "Indicators for human review, not proof of fraud. Synthetic demonstration data.",
    }


def portfolio_context(ws: Workspace, incidents: pd.DataFrame, top_n: int = 15) -> dict:
    inc = incidents
    by_type = inc.groupby("incident_type").agg(cases=("incident_id", "size"), potential_exposure=("potential_exposure", "sum"),
                                               confirmed_loss=("confirmed_loss", "sum")).round(2)
    sup = ws.suppliers.head(10)
    return {
        "scope": "portfolio",
        "totals": {"cases": len(inc), "potential_exposure_usd": round(float(inc.potential_exposure.sum()), 2),
                   "already_paid_usd": round(float(inc.paid_exposure.sum()), 2),
                   "confirmed_loss_usd": round(float(inc.confirmed_loss.sum()), 2),
                   "cases_by_band": inc.band.value_counts().to_dict(),
                   "cases_raised_only_by_anomaly_model": int((inc.detection_methods == "ISOLATION_FOREST").sum()),
                   "invoice_lines_pending_receipt_in_grace_period": len(ws.pending),
                   "invoice_lines_analysed": ws.summary.get("inputs", {}).get("invoices"),
                   "data_as_of": ws.summary.get("as_of_date")},
        "by_case_type": {TYPE_LABEL.get(k, k): v for k, v in by_type.to_dict(orient="index").items()},
        "top_cases_in_queue_order": _df_records(inc.head(top_n), ["incident_id", "incident_type", "supplier_id",
                                                "potential_exposure", "paid_exposure", "confirmed_loss", "band",
                                                "risk_score", "evidence_tier"], top_n),
        "top_suppliers_by_exposure": _df_records(sup, ["supplier_id", "supplier_name", "incidents", "spend",
                                                       "potential_exposure", "confirmed_loss",
                                                       "exposure_share_of_spend"], 10),
        "how_cases_are_ranked": "Default queue order = expected value (potential exposure x rule-confidence factor "
                                "HIGH 1.0 / MEDIUM 0.6 / LOW 0.25). Risk score 0-100 = exposure (35) + evidence "
                                "strength (30) + money already paid (15) + anomaly model (20); bands P1 >= 60, P2 >= 35.",
        "thresholds_applied": thresholds(ws),
        "disclaimer": "Indicators for human review, not proof of fraud. Synthetic demonstration data.",
    }


# ------------------------------------------------------------------ providers
def _secret(name: str, secrets=None) -> str | None:
    v = os.environ.get(name)
    if v:
        return v.strip()
    try:
        if secrets is not None and name in secrets:
            return str(secrets[name]).strip()
    except Exception:  # noqa: BLE001 - streamlit raises if no secrets file exists
        return None
    return None


@dataclass
class ProviderConfig:
    name: str | None
    model: str | None
    key: str | None = field(default=None, repr=False)

    @property
    def online(self) -> bool:
        return bool(self.name and self.key)

    @property
    def label(self) -> str:
        names = {"openai": "OpenAI", "groq": "Groq", "gemini": "Google Gemini"}
        return f"{names.get(self.name, self.name)} · {self.model}" if self.online else "Offline copilot (no network needed)"


def resolve_provider(secrets=None, force_offline: bool = False) -> ProviderConfig:
    if force_offline or (_secret("LEAK_HUNTER_COPILOT_OFFLINE", secrets) or "").lower() in ("1", "true", "yes"):
        return ProviderConfig(None, None)
    pref = (_secret("LEAK_HUNTER_LLM_PROVIDER", secrets) or "").lower()
    order = [pref] + [p for p in PROVIDERS if p != pref] if pref in PROVIDERS else list(PROVIDERS)
    for name in order:
        spec = PROVIDERS[name]
        key = next((k for k in (_secret(n, secrets) for n in spec["key"]) if k), None)
        if key:
            return ProviderConfig(name, _secret(spec["model_env"], secrets) or spec["default_model"], key)
    return ProviderConfig(None, None)


class ProviderError(RuntimeError):
    pass


def call_llm(cfg: ProviderConfig, system: str, messages: list[dict], timeout: float = 25.0,
             max_tokens: int = 900) -> str:
    """messages: [{"role": "user"|"assistant", "content": str}, ...]. Raises ProviderError."""
    import requests
    spec = PROVIDERS[cfg.name]
    try:
        if cfg.name in ("openai", "groq"):
            body = {"model": cfg.model, "temperature": 0.1, "max_tokens": max_tokens,
                    "messages": [{"role": "system", "content": system}] + messages}
            r = requests.post(spec["url"], json=body, timeout=timeout,
                              headers={"Authorization": f"Bearer {cfg.key}", "Content-Type": "application/json"})
            if r.status_code != 200:
                raise ProviderError(f"{cfg.name} HTTP {r.status_code}: {r.text[:200]}")
            return r.json()["choices"][0]["message"]["content"].strip()
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                             for m in messages],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": max_tokens}}
        r = requests.post(spec["url"].format(model=cfg.model), json=body, timeout=timeout,
                          headers={"x-goog-api-key": cfg.key, "Content-Type": "application/json"})
        if r.status_code != 200:
            raise ProviderError(f"gemini HTTP {r.status_code}: {r.text[:200]}")
        parts = r.json()["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()
    except ProviderError:
        raise
    except Exception as e:  # noqa: BLE001 - network errors, timeouts, malformed JSON
        raise ProviderError(f"{cfg.name}: {type(e).__name__}: {e}") from e


# ------------------------------------------------------------------ citation check
def _amounts_in(obj) -> set[float]:
    out: set[float] = set()
    if isinstance(obj, dict):
        for v in obj.values():
            out |= _amounts_in(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= _amounts_in(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.add(round(float(obj), 2))
    elif isinstance(obj, str):
        out |= {round(float(m.replace(",", "")), 2) for m in AMOUNT_RE.findall(obj)}
    return out


def verify_citations(answer: str, context: dict) -> dict:
    """IDs and dollar amounts in ``answer`` that do not appear in ``context``."""
    blob = json.dumps(context, default=str)
    ids = set(ID_RE.findall(answer))
    unknown_ids = sorted(i for i in ids if i not in blob)
    known_amounts = _amounts_in(context)
    amts = {round(float(a.replace(",", "")), 2) for a in AMOUNT_RE.findall(answer)}
    unknown_amts = sorted(a for a in amts if a not in known_amounts and a != 0)
    return {"ids_cited": sorted(ids), "amounts_cited": sorted(amts), "unverified_ids": unknown_ids,
            "unverified_amounts": unknown_amts}


# ------------------------------------------------------------------ offline copilot
def _intent(q: str, scope: str) -> str:
    s = q.lower()
    table = ([("fraud", ["fraud", "steal", "theft", "criminal", "guilty", "intent", "corrupt", "kickback"]),
              ("letter", ["letter", "email", "inquiry", "enquiry", "write to", "draft"]),
              ("evidence", ["evidence", "prove", "proof", "why", "overcharge", "duplicate", "support"]),
              ("next", ["next", "step", "verify", "check", "what should", "action", "do now"]),
              ("supplier", ["supplier", "vendor"]), ("linked", ["linked", "related", "connected", "other case"]),
              ("timeline", ["timeline", "when", "date", "history", "sequence"]),
              ("score", ["score", "rank", "priority", "band", "points"]),
              ("summary", ["summar", "risk", "calculat", "exposure", "how much", "money", "amount", "overview"])]
             if scope == "case" else
             [("fraud", ["fraud", "steal", "theft", "criminal", "corrupt", "kickback"]),
              ("top", ["first", "top", "priorit", "investigate", "queue", "which case", "worst"]),
              ("money", ["money", "exposure", "paid", "loss", "how much", "total", "risk"]),
              ("suppliers", ["supplier", "vendor"]), ("types", ["type", "kind", "category", "breakdown"]),
              ("method", ["how", "rank", "score", "method", "model", "rule", "work"])])
    for name, words in table:
        if any(w in s for w in words):
            return name
    return "help"


def _case_summary(c: dict) -> str:
    k = c["case"]
    rs = k["risk_score"]
    L = [f"**{c['case']['case_id']}** — {', '.join(k['finding_types'])} involving supplier **{k['supplier_id']}** "
         f"(queue rank {k['queue_rank']} of {k['cases_in_queue']}, band {k['band']}).", "",
         f"- **Potential exposure:** {usd(k['potential_exposure_usd'])} · **already paid:** {usd(k['already_paid_usd'])} "
         f"· **confirmed loss:** {usd(k['confirmed_loss_usd'])}",
         f"- **Risk score {rs['total']:.1f}/100:** exposure {rs['exposure_points_of_35']:.1f}/35, evidence "
         f"{rs['evidence_points_of_30']:.1f}/30, money paid {rs['money_paid_points_of_15']:.1f}/15, anomaly model "
         f"{rs['anomaly_model_points_of_20']:.1f}/20.", "", "**How the exposure was calculated**"]
    for f in c["findings"]:
        if f["method"] != "business rule":
            continue
        L.append(f"- *{f['type']}* on {f['invoice_line_id']}:")
        L += [f"  - {s}" for s in f["calculation_steps"]]
    if len(c["exposure_breakdown_per_line"]) and len(c["findings"]) > 1:
        L.append(f"- Overlap rule: {c['thresholds_applied']['exposure_overlap_rule']} — total counted once: "
                 f"{usd(k['potential_exposure_usd'])}.")
    if k["confirmed_loss_usd"] == 0:
        L.append("\nNo loss is confirmed yet: the amount is a formula-based estimate until the records below are verified.")
    return "\n".join(L)


def _case_evidence(c: dict) -> str:
    th = c["thresholds_applied"]
    L = [f"Evidence for **{c['case']['case_id']}** (evidence tier: {c['case']['evidence_tier']}, rule confidence "
         f"{c['case']['rule_confidence']}):", ""]
    for f in c["findings"]:
        d = f["details"]
        t = f["type"]
        if t == "Duplicate invoice":
            docs = d.get("group_documents", [])
            L.append(f"**Duplicate invoice ({d.get('match_type', '').lower()} match)** — supplier invoice number(s) "
                     f"{', '.join(d.get('group_invoice_numbers', []))} entered as " +
                     "; ".join(f"{x['invoice_id']} on {x['invoice_date']} ({usd(x['document_amount'])}, paid "
                               f"{usd(x['amount_paid'])}, {x['payment_status']})" for x in docs) + ".")
            if d.get("role") == "REPEAT":
                L.append(f"- Payment records: realized duplicate payment {usd(d.get('realized_duplicate_payment'))} "
                         f"({d.get('payment_assessment', '').replace('_', ' ').lower()}).")
            L.append(f"- Rule: {th['duplicate_matching']}.")
        elif t == "PO price mismatch":
            L.append(f"**PO price mismatch** on {f['invoice_line_id']} (PO line {f['po_line_id']}): invoiced "
                     f"{usd(d.get('invoice_unit_price'))}/unit vs approved {usd(d.get('approved_unit_price'))}/unit "
                     f"(+{d.get('price_difference_pct')}%) on {d.get('quantity_invoiced'):g} units.")
            if d.get("applied_price_amendment_id"):
                L.append(f"- Approved amendment {d['applied_price_amendment_id']} was applied before comparing.")
            if d.get("billed_at_unapproved_amendment_price"):
                L.append(f"- The invoice uses the price of non-approved amendment(s) "
                         f"{', '.join(d['billed_at_unapproved_amendment_price'])}.")
            if d.get("ambiguity"):
                L.append(f"- Caution: {d['ambiguity'].replace('_', ' ').lower()} (confidence {d.get('confidence')}).")
            L.append(f"- Threshold: {th['price_mismatch_materiality']}.")
        elif t == "Quantity discrepancy":
            L.append(f"**Quantity discrepancy** on PO line {f['po_line_id']}: invoiced {d.get('cumulative_invoiced'):g} "
                     f"units in total vs {d.get('supported_quantity'):g} supported (received "
                     f"{d.get('received_total_all_receipts'):g} across receipts {', '.join(d.get('receipt_ids', [])) or 'none'}; "
                     f"approved quantity {d.get('approved_quantity'):g}). Unsupported units on this line: "
                     f"{d.get('excess_quantity'):g}.")
            L.append(f"- Threshold: {th['quantity_materiality']}; grace period {th['posting_lag_grace_period_days']} days "
                     f"(invoice age {d.get('invoice_age_days')} days).")
        elif t == "Historical price spike":
            L.append(f"**Historical price spike** on {f['invoice_line_id']}: {usd(d.get('invoice_unit_price'))} is "
                     f"{d.get('price_ratio')}x the median {usd(d.get('baseline_unit_price'))} of "
                     f"{d.get('n_prior_observations')} earlier purchases (range {usd(d.get('prior_min'))}–"
                     f"{usd(d.get('prior_max'))}); threshold {d.get('effective_ratio_threshold')}x.")
            L.append("- This alone is an unusual price, not a contractual breach." if not d.get("has_contractual_basis")
                     else f"- The approved PO price {usd(d.get('approved_unit_price'))} is also exceeded.")
        elif f["method"] == "anomaly model":
            L.append(f"**Anomaly model:** score {d.get('anomaly_score', 0):.3f} (rank {d.get('risk_rank')}); drivers: " +
                     ", ".join(f"{x['feature']} (z {x['robust_z']})" for x in d.get("top_features", [])) +
                     ". Statistical rarity only — no money attached.")
        L.append("")
    L.append("**What is proven vs. not yet proven:** the records above prove the discrepancy against the PO, receipts "
             "and payment history in the system. They do not show the supplier's intent, credit notes issued outside "
             "these records, or unposted deliveries — see the next steps.")
    return "\n".join(L)


def _case_letter(c: dict) -> str:
    k = c["case"]
    sp = (c["supplier_profile"] or [{}])[0]
    name = sp.get("supplier_name") or k["supplier_id"]
    rows, asks = [], []
    priced = {f["invoice_line_id"] for f in c["findings"] if f["type"] == "PO price mismatch"}
    for f in c["findings"]:
        if f["type"] == "Historical price spike" and f["invoice_line_id"] in priced:
            continue   # same overcharge already raised as a PO price mismatch
        if f["method"] != "business rule" or f["potential_exposure_usd"] == 0 and f["type"] != "Duplicate invoice":
            continue
        d = f["details"]
        if f["type"] == "Duplicate invoice" and d.get("role") == "REPEAT":
            rows.append(f"- Invoice no. {d.get('invoice_number')} appears to have been submitted twice "
                        f"(our references {', '.join(x['invoice_id'] for x in d.get('group_documents', []))}), "
                        f"amount {usd(f['line_amount_usd'])}.")
            asks.append("confirm whether invoice no. {} was issued once or twice and, if paid twice, issue a credit note "
                        "for the duplicate amount".format(d.get("invoice_number")))
        elif f["type"] == "PO price mismatch":
            rows.append(f"- Invoice no. {f['supplier_invoice_number']} (line {f['invoice_line_id']}): unit price "
                        f"{usd(d.get('invoice_unit_price'))} vs agreed {usd(d.get('approved_unit_price'))} on purchase order "
                        f"line {f['po_line_id']} — difference {usd(f['potential_exposure_usd'])}.")
            asks.append("provide the contract or approved amendment supporting the invoiced price, or a credit note")
        elif f["type"] == "Quantity discrepancy":
            rows.append(f"- Purchase order line {f['po_line_id']}: {d.get('cumulative_invoiced'):g} units invoiced, "
                        f"{d.get('supported_quantity'):g} units received per our records (invoice no. "
                        f"{f['supplier_invoice_number']}) — difference valued at {usd(f['potential_exposure_usd'])}.")
            asks.append("provide proof of delivery (signed delivery notes) for the additional units, or a credit note")
        elif f["type"] == "Historical price spike":
            rows.append(f"- Invoice no. {f['supplier_invoice_number']}: unit price {usd(d.get('invoice_unit_price'))} for "
                        f"{f['item_id']}, compared with {usd(d.get('baseline_unit_price'))} on recent orders.")
            asks.append("explain the basis of the price change for this item")
    if not rows:
        return ("This case has no contractual discrepancy to raise with the supplier (it is a statistical or "
                "historical-price indicator only). I recommend internal review before contacting the supplier.")
    asks = list(dict.fromkeys(asks))
    return "\n".join([
        f"**Subject:** Request for clarification — invoices from {name} ({k['supplier_id']})", "",
        f"Dear {name} accounts receivable team,", "",
        "As part of our routine accounts payable review, we have identified the following items that we cannot "
        "reconcile with our purchase orders and receiving records:", ""] + rows + ["",
        "To help us close this review, could you please " + "; ".join(asks) + "?", "",
        "Please send the requested documents within [15] business days to [email address], quoting reference "
        f"{k['case_id']}. If you believe our records are incomplete, we would welcome copies of any supporting "
        "documentation. This request does not imply any wrongdoing; it is part of our standard reconciliation.", "",
        "Kind regards,  ", "[Your name]  ", "[Title], [Organization]"])


def _case_next(c: dict) -> str:
    steps = c["recommended_next_steps"]
    return "Recommended verification steps, generated from this case's evidence:\n\n" + "\n".join(
        f"{i}. **{s['step']}** — {s['why']}" + (f" (records: {', '.join(map(str, s['records']))})" if s["records"] else "")
        for i, s in enumerate(steps, 1))


def _case_supplier(c: dict) -> str:
    if not c["supplier_profile"]:
        return "The evidence available to me does not include a supplier profile for this case."
    s = c["supplier_profile"][0]
    return (f"**{s.get('supplier_name')} ({s.get('supplier_id')})** — {s.get('category')}, {s.get('country')}, "
            f"{s.get('supplier_tier')} supplier.\n\n- {s.get('invoices')} invoices / {s.get('invoice_lines')} lines, spend "
            f"{usd(s.get('spend'))}\n- {int(s.get('incidents') or 0)} cases ({int(s.get('p1_incidents') or 0)} P1) in "
            f"{int(s.get('months_with_incidents') or 0)} different months; potential exposure "
            f"{usd(s.get('potential_exposure'))} ({100 * float(s.get('exposure_share_of_spend') or 0):.1f}% of spend); "
            f"confirmed loss {usd(s.get('confirmed_loss'))}\n- Case types: "
            f"{str(s.get('incident_types', '')).replace('|', ', ').replace('_', ' ').lower()}\n\n"
            "A supplier profile shows concentration, not wrongdoing.")


def _case_linked(c: dict) -> str:
    if not c["linked_cases"]:
        return "No other case shares a PO line or invoice with this one."
    return "Cases sharing a PO line or invoice with this one:\n\n" + "\n".join(
        f"- {x['incident_id']}: {TYPE_LABEL.get(x['incident_type'], x['incident_type'])}, exposure "
        f"{usd(x['potential_exposure'])}, band {x['band']}" for x in c["linked_cases"])


def _case_timeline(c: dict) -> str:
    return "Sequence of records for this case:\n\n" + "\n".join(
        f"- {e['date'][:10]} — {e['event']} {e['record_id']}: {e['detail']}" for e in c["timeline"])


def _case_score(c: dict) -> str:
    k = c["case"]
    rs = k["risk_score"]
    return (f"Queue rank **{k['queue_rank']}** of {k['cases_in_queue']} (default order: expected value "
            f"{usd(k['expected_value_usd'])} = exposure x rule-confidence factor). Band **{k['band']}**.\n\n"
            f"Risk score {rs['total']:.1f}/100 = exposure {rs['exposure_points_of_35']:.1f}/35 + evidence "
            f"{rs['evidence_points_of_30']:.1f}/30 ({k['rule_confidence']} confidence) + money already paid "
            f"{rs['money_paid_points_of_15']:.1f}/15 + anomaly model {rs['anomaly_model_points_of_20']:.1f}/20"
            + (f" (percentile {rs['anomaly_model_percentile']:.3f} of training-period scores)"
               if rs["anomaly_model_percentile"] is not None else "") + ".")


def _fraud(_c: dict) -> str:
    return ("The system cannot determine fraud or intent. It reports discrepancies and their potential financial "
            "impact; whether they are errors, legitimate exceptions or misconduct must be established by an "
            "investigator from the supplier's response and the underlying documents.")


def _p_top(c: dict) -> str:
    return ("Investigate in queue order (expected value first):\n\n" + "\n".join(
        f"{i}. **{x['incident_id']}** — {TYPE_LABEL.get(x['incident_type'], x['incident_type'])}, {x['supplier_id']}, "
        f"exposure {usd(x['potential_exposure'])} (paid {usd(x['paid_exposure'])}, confirmed loss "
        f"{usd(x['confirmed_loss'])}), band {x['band']}, score {x['risk_score']:.0f}"
        for i, x in enumerate(c["top_cases_in_queue_order"][:10], 1)) + f"\n\n{c['how_cases_are_ranked']}")


def _p_money(c: dict) -> str:
    t = c["totals"]
    return (f"Across **{t['cases']}** cases: potential exposure **{usd(t['potential_exposure_usd'])}**, of which "
            f"**{usd(t['already_paid_usd'])}** is already paid out; confirmed loss (duplicate payments proven by "
            f"payment records) **{usd(t['confirmed_loss_usd'])}**. {t['invoice_lines_pending_receipt_in_grace_period']} "
            f"invoice line(s) are billed ahead of receipts but still inside the grace period. Bands: "
            f"{', '.join(f'{k} {v}' for k, v in sorted(t['cases_by_band'].items()))}.\n\nBy case type:\n" + "\n".join(
                f"- {k}: {v['cases']} cases, {usd(v['potential_exposure'])} potential, {usd(v['confirmed_loss'])} confirmed"
                for k, v in c["by_case_type"].items()))


def _p_suppliers(c: dict) -> str:
    return "Suppliers with the most potential exposure:\n\n" + "\n".join(
        f"{i}. **{s['supplier_id']}** {s['supplier_name']} — {usd(s['potential_exposure'])} across "
        f"{int(s['incidents'])} cases ({100 * float(s['exposure_share_of_spend']):.1f}% of spend {usd(s['spend'])}); "
        f"confirmed loss {usd(s['confirmed_loss'])}" for i, s in enumerate(c["top_suppliers_by_exposure"][:8], 1)) + \
        "\n\nConcentration is a navigation aid, not evidence of wrongdoing by itself."


def _p_method(c: dict) -> str:
    th = c["thresholds_applied"]
    return (f"{c['how_cases_are_ranked']}\n\nRules and thresholds:\n" + "\n".join(f"- {k.replace('_', ' ')}: {v}"
                                                                                 for k, v in th.items()))


CASE_HANDLERS = {"summary": _case_summary, "evidence": _case_evidence, "letter": _case_letter, "next": _case_next,
                 "supplier": _case_supplier, "linked": _case_linked, "timeline": _case_timeline, "score": _case_score,
                 "fraud": _fraud}
PORTFOLIO_HANDLERS = {"top": _p_top, "money": _p_money, "suppliers": _p_suppliers, "types": _p_money,
                      "method": _p_method, "fraud": _fraud}


def offline_answer(question: str, context: dict) -> str:
    scope = context.get("scope", "case")
    intent = _intent(question, scope)
    handlers = CASE_HANDLERS if scope == "case" else PORTFOLIO_HANDLERS
    if intent in handlers:
        return handlers[intent](context)
    if scope == "case":
        return ("I answer from this case's evidence only. Try: *summarize the risk*, *what evidence supports it*, "
                "*draft a letter to the supplier*, *next steps*, *supplier profile*, *linked cases*, *timeline*, "
                "*why this rank*.\n\n" + _case_summary(context))
    return ("I answer from the case queue only. Try: *which cases first*, *how much money is at risk*, *which "
            "suppliers*, *how are cases ranked*.\n\n" + _p_money(context))


# ------------------------------------------------------------------ orchestration
@dataclass
class CopilotReply:
    text: str
    mode: str                 # "llm" | "offline" | "offline-fallback"
    provider: str
    latency_s: float
    citations: dict = field(default_factory=dict)
    error: str | None = None


def ask(question: str, context: dict, history: list[dict] | None = None, provider: ProviderConfig | None = None,
        timeout: float = 25.0) -> CopilotReply:
    """Answer ``question`` grounded on ``context``. Never raises for provider problems."""
    t0 = time.time()
    provider = provider or ProviderConfig(None, None)
    if provider.online:
        evidence = json.dumps(context, default=str, separators=(",", ":"))
        system = f"{SYSTEM_PROMPT}\n\n<evidence>\n{evidence}\n</evidence>"
        msgs = [m for m in (history or [])[-6:] if m["role"] in ("user", "assistant")] + [{"role": "user",
                                                                                         "content": question}]
        try:
            text = call_llm(provider, system, msgs, timeout=timeout)
            if not text:
                raise ProviderError("empty response")
            return CopilotReply(text, "llm", provider.label, round(time.time() - t0, 2), verify_citations(text, context))
        except ProviderError as e:
            text = offline_answer(question, context)
            return CopilotReply(text, "offline-fallback", "Offline copilot", round(time.time() - t0, 2),
                                verify_citations(text, context), error=str(e))
    text = offline_answer(question, context)
    return CopilotReply(text, "offline", "Offline copilot", round(time.time() - t0, 2), verify_citations(text, context))
