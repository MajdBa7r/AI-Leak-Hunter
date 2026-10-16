"""AI Leak Hunter - procurement leakage investigation app.

    streamlit run app/streamlit_app.py

Reads the detection outputs (default ./outputs, or $LEAK_HUNTER_OUT_DIR) and the
procurement CSVs. Never reads ground truth, except on the Evaluation page, which
only displays the saved benchmark reports. Decisions go to an append-only SQLite
log (outputs/case_decisions.sqlite, or $LEAK_HUNTER_CASE_DB).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import charts  # noqa: E402
from leak_hunter import copilot as cp  # noqa: E402
from leak_hunter.case_store import STATUSES, CaseStore, default_db_path  # noqa: E402
from leak_hunter.investigation import DISCLAIMER, TYPE_LABEL, Workspace, usd  # noqa: E402

OUT_DIR = Path(os.environ.get("LEAK_HUNTER_OUT_DIR", ROOT / "outputs"))
PAGES = ["Overview", "Case queue", "Case file", "Suppliers", "Decision log", "Evaluation"]
ORDERS = {"Money first (expected value)": "value_rank", "Risk score": "risk_rank", "Severity label": "severity_rank"}
STATUS_TEXT = {"NEW": "New", "IN_REVIEW": "In review", "NEEDS_INFORMATION": "Needs information",
               "CONFIRMED_LEAKAGE": "Confirmed leakage", "NO_ISSUE": "No issue (explained)", "ESCALATED": "Escalated"}

st.set_page_config(page_title="AI Leak Hunter", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
.case-head {border-left: 6px solid var(--band); padding: .2rem 0 .2rem 1rem; margin-bottom: .6rem;}
.case-head h2 {margin: 0; font-size: 1.7rem;}
.case-head p {margin: .2rem 0 0 0; color: #5B6676;}
.note {color: #5B6676; font-size: .9rem;}
.step {padding: .45rem .7rem; border: 1px solid #D5DBE3; border-radius: 6px; background: #fff; margin-bottom: .4rem;}
.step b {color: #1E2733;}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data
@st.cache_resource(show_spinner="Loading detection outputs...")
def load_workspace(out_dir: str, stamp: float) -> Workspace:
    return Workspace.load(out_dir)


def outputs_stamp() -> float:
    p = OUT_DIR / "run_summary.json"
    return p.stat().st_mtime if p.exists() else 0.0


def run_pipeline():
    from leak_hunter.pipeline import run_detection, save_outputs
    with st.spinner("Running detection (rules + Isolation Forest)..."):
        save_outputs(run_detection(), OUT_DIR)
    load_workspace.clear()


@st.cache_resource
def store(db_path: str) -> CaseStore:
    return CaseStore(db_path)


def md(text: str) -> str:
    """Escape '$' so Streamlit markdown does not read amounts as LaTeX."""
    return str(text).replace("$", "\\$")


def open_case(case_id: str):
    st.session_state["case_id"] = case_id
    st.session_state["page"] = "Case file"


if not (OUT_DIR / "run_summary.json").exists():
    # first start (e.g. a fresh Streamlit Cloud deploy): build the case queue from the repository's CSVs
    st.title("AI Leak Hunter")
    try:
        run_pipeline()
        st.rerun()
    except Exception as err:  # noqa: BLE001
        st.error(f"Could not run detection automatically: {err}")
        if st.button("Try again", type="primary"):
            st.rerun()
        st.stop()

ws = load_workspace(str(OUT_DIR), outputs_stamp())
cs = store(str(default_db_path(OUT_DIR)))
inc = ws.incidents.copy()
inc["review_status"] = cs.status_of(inc.incident_id).values

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### AI Leak Hunter")
    st.caption("Procurement leakage investigation")
    page = st.radio("View", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    reviewer = st.text_input("Your name (for decisions)", key="reviewer", placeholder="e.g. M. Bakr")
    offline_only = st.toggle("Copilot: offline mode only", key="offline_only",
                             help="Answers come from built-in templates over the case evidence; no network calls.")
    provider = cp.resolve_provider(st.secrets, force_offline=offline_only)
    st.caption(f"Copilot: {provider.label}")
    order_label = st.selectbox("Queue order", list(ORDERS), key="order",
                               help="Money first was chosen by a pre-committed rule on a fresh replica dataset; "
                                    "Risk score is more precise under noisy data. See Evaluation.")
    s = ws.summary
    st.caption(f"Data as of {s.get('as_of_date')} · {s['inputs']['invoices']:,} invoice lines · "
               f"run took {s.get('runtime_seconds')} s")
    if st.button("Re-run detection", help="Re-reads the procurement CSVs and rebuilds all cases"):
        run_pipeline()
        st.rerun()

inc = inc.sort_values(ORDERS[order_label]).reset_index(drop=True)
inc.insert(0, "queue_position", range(1, len(inc) + 1))


def render_copilot(scope_key: str, context: dict, prompts: dict, intro: str):
    """Grounded chat: quick prompts + chat input; history kept per case / page in session state."""
    hist_key = f"chat_{scope_key}"
    history = st.session_state.setdefault(hist_key, [])
    st.caption(f"{intro} · {provider.label}. Answers cite record IDs and amounts from the evidence; the copilot does "
               "not judge fraud.")

    def submit(question: str):
        reply = cp.ask(question, context, [{"role": m["role"], "content": m["content"]} for m in history],
                       provider)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": reply.text, "mode": reply.mode, "provider": reply.provider,
                        "latency": reply.latency_s, "citations": reply.citations, "error": reply.error})

    def on_pick():
        picked = st.session_state.get(f"qp_{scope_key}")
        if picked:
            submit(picked)
            st.session_state[f"qp_{scope_key}"] = None     # chips act as one-shot buttons

    st.pills("Quick questions", list(prompts.values()), key=f"qp_{scope_key}", on_change=on_pick,
             label_visibility="collapsed")
    # newest exchange first, so the answer to the question just asked is always in view
    pairs = [history[i:i + 2] for i in range(0, len(history), 2)][::-1]
    for pair in pairs:
        for m in pair:
            with st.chat_message(m["role"], avatar="🧑‍💼" if m["role"] == "user" else "🔎"):
                st.markdown(md(m["content"]))
                if m["role"] == "assistant":
                    c = m.get("citations") or {}
                    note = (f"{'Model' if m['mode'] == 'llm' else 'Offline copilot'} · {m['latency']} s · "
                            f"{len(c.get('ids_cited', []))} IDs and {len(c.get('amounts_cited', []))} amounts cited")
                    bad = c.get("unverified_ids", []) + [usd(a) for a in c.get("unverified_amounts", [])]
                    st.caption(md(note + (f" · not found in the evidence (verify): {', '.join(map(str, bad))}" if bad
                                          else " · all found in the evidence")))
                    if m.get("error"):
                        st.caption(f"Model unavailable ({m['error'][:120]}); answered offline from the same evidence.")
    question = st.chat_input("Ask about this case's evidence..." if scope_key != "overview" else
                             "Ask about the case queue...", key=f"input_{scope_key}")
    if question:
        with st.spinner("Reading the evidence..."):
            submit(question)
        st.rerun()
    c1, c2 = st.columns([5, 1])
    with c1.expander("Evidence the copilot can see (JSON)"):
        st.json(context, expanded=False)
    if c2.button("Clear chat", key=f"clear_{scope_key}", width="stretch", disabled=not history):
        history.clear()
        st.rerun()


def queue_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "#": df.queue_position, "Band": df.band, "Case": df.incident_id,
        "Type": df.incident_type.map(TYPE_LABEL), "Supplier": df.supplier_id,
        "Potential exposure": df.potential_exposure, "Already paid": df.paid_exposure,
        "Confirmed loss": df.confirmed_loss, "Risk score": df.risk_score,
        "Evidence": df.evidence_tier.str.replace("_", " ").str.capitalize(),
        "Detected by": df.detection_methods.str.replace("RULE_BASED", "Rules").str.replace("ISOLATION_FOREST", "ML")
                         .str.replace("|", " + ", regex=False),
        "Status": df.review_status.map(STATUS_TEXT)})


MONEY = st.column_config.NumberColumn(format="dollar")
QCFG = {"Potential exposure": MONEY, "Already paid": MONEY, "Confirmed loss": MONEY,
        "Risk score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f")}

# ------------------------------------------------------------------ overview
if page == "Overview":
    st.title("Leakage overview")
    st.caption(DISCLAIMER)
    pend = float(ws.pending.amount_at_stake_if_unreceived.sum()) if len(ws.pending) else 0.0
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(charts.money_states(inc.potential_exposure.sum(), inc.paid_exposure.sum(),
                                            inc.confirmed_loss.sum(), pend), width="stretch")
    with right:
        b = inc.band.value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("Investigate now (P1)", int(b.get("P1", 0)))
        c2.metric("This cycle (P2)", int(b.get("P2", 0)))
        c3.metric("Monitor (P3)", int(b.get("P3", 0)))
        summ = cs.summary(inc)
        c1, c2, c3 = st.columns(3)
        c1.metric("Cases reviewed", f"{summ['reviewed']} / {summ['cases']}")
        c2.metric("Confirmed rate",
                  "—" if summ["reviewer_confirmed_precision"] is None else f"{summ['reviewer_confirmed_precision']:.0%}",
                  help="Confirmed leakage / closed cases, from the decision log")
        c3.metric("Recovered", usd(summ["recovered_amount"]))
        ml_only = int((inc.detection_methods == "ISOLATION_FOREST").sum())
        st.markdown(f"<p class='note'>{len(inc)} cases: {len(inc) - ml_only} raised by business rules, {ml_only} raised "
                    f"only by the statistical model (no monetary exposure). {len(ws.pending)} invoice "
                    f"line{'s are' if len(ws.pending) != 1 else ' is'} billed ahead of posted receipts and "
                    f"{'wait' if len(ws.pending) != 1 else 'waits'} inside the grace period.</p>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.exposure_by_type(inc), width="stretch")
    c2.plotly_chart(charts.exposure_by_month(inc, ws.recon), width="stretch")
    st.subheader("Top of the queue")
    st.dataframe(queue_table(inc.head(10)), hide_index=True, column_config=QCFG, width="stretch")
    st.subheader("Investigation copilot")
    render_copilot("overview", cp.portfolio_context(ws, inc), cp.OVERVIEW_PROMPTS,
                   "Grounded on the whole case queue")

# ------------------------------------------------------------------ queue
elif page == "Case queue":
    st.title("Case queue")
    f1, f2, f3, f4 = st.columns([1.3, 2, 1.6, 1.4])
    bands = f1.pills("Band", ["P1", "P2", "P3"], default=["P1", "P2", "P3"], selection_mode="multi") or []
    types = f2.multiselect("Type", list(TYPE_LABEL), format_func=TYPE_LABEL.get)
    sup = f3.multiselect("Supplier", sorted(inc.supplier_id.unique()))
    stat = f4.multiselect("Status", STATUSES, format_func=STATUS_TEXT.get)
    g1, g2, g3 = st.columns([1.2, 2, 3])
    min_exp = g1.number_input("Minimum exposure ($)", min_value=0.0, value=0.0, step=100.0)
    tiers = g2.segmented_control("Evidence", ["All", "Rule violations", "Unusual patterns only"], default="All") or "All"
    q = g3.text_input("Search case, invoice or line ID")
    v = inc[inc.band.isin(bands) & (inc.potential_exposure >= min_exp)]
    if types:
        v = v[v.finding_types.map(lambda s: any(t in s.split("|") for t in types))]
    if sup:
        v = v[v.supplier_id.isin(sup)]
    if stat:
        v = v[v.review_status.isin(stat)]
    if tiers != "All":
        v = v[v.evidence_tier == ("RULE_VIOLATION" if tiers == "Rule violations" else "UNUSUAL_PATTERN")]
    if q:
        qq = q.strip().upper()
        v = v[v.incident_id.str.upper().str.contains(qq, regex=False) | v.invoice_line_ids.str.upper().str.contains(qq, regex=False)]
    st.caption(f"{len(v)} cases · {usd(v.potential_exposure.sum())} potential exposure · ordered by {order_label.lower()}. "
               "Select a row to open its case file.")
    tbl = queue_table(v)

    def _on_select():
        sel = st.session_state["queue_sel"]["selection"]["rows"]
        if sel:
            open_case(tbl.iloc[sel[0]]["Case"])

    st.dataframe(tbl, hide_index=True, column_config=QCFG, width="stretch", height=560, key="queue_sel",
                 on_select=_on_select, selection_mode="single-row")
    st.download_button("Download this queue (CSV)", tbl.to_csv(index=False), "leak_hunter_queue.csv", "text/csv")
    if len(ws.pending):
        n_p = len(ws.pending)
        with st.expander(f"Awaiting receipts: {n_p} line{'s' if n_p != 1 else ''} inside the grace period (not cases yet)"):
            st.dataframe(ws.pending, hide_index=True, width="stretch")

# ------------------------------------------------------------------ case file
elif page == "Case file":
    ids = inc.incident_id.tolist()
    cur = st.session_state.get("case_id", ids[0])
    cur = cur if cur in ids else ids[0]
    case_id = st.selectbox("Case", ids, index=ids.index(cur),
                           format_func=lambda i: f"#{int(inc.loc[inc.incident_id == i, 'queue_position'].iloc[0])}  {i}")
    st.session_state["case_id"] = case_id
    row = inc[inc.incident_id == case_id].iloc[0]
    findings = ws.case_findings(case_id)
    color = charts.BAND_COLORS[row.band]
    types = " + ".join(TYPE_LABEL[t] for t in row.finding_types.split("|"))
    st.markdown(f"<div class='case-head' style='--band:{color}'><h2>{types}</h2>"
                f"<p>{case_id} · supplier {row.supplier_id} · {row.band} {row.band_label.lower()} · queue position "
                f"{int(row.queue_position)} · status {STATUS_TEXT[row.review_status].lower()}</p></div>",
                unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Potential exposure", usd(row.potential_exposure))
    m2.metric("Already paid", usd(row.paid_exposure))
    m3.metric("Confirmed loss", usd(row.confirmed_loss), help="Only duplicate payments proven by payment records")
    m4.metric("Evidence", row.evidence_tier.replace("_", " ").capitalize())
    st.plotly_chart(charts.risk_breakdown(row), width="stretch")

    tab_copilot, tab_why, tab_map, tab_records, tab_next, tab_decide = st.tabs(
        ["Copilot", "Why it was flagged", "Evidence map", "Records", "Next steps", "Decision & report"])
    with tab_copilot:
        render_copilot(case_id, cp.case_context(ws, case_id), cp.QUICK_PROMPTS,
                       f"Grounded on the evidence of {case_id}")
    with tab_why:
        for _, f in findings.iterrows():
            with st.container(border=True):
                st.markdown(f"**{TYPE_LABEL[f.finding_type]}** · {f.invoice_line_id} · "
                            f"{'business rule' if f.detection_method == 'RULE_BASED' else 'statistical model'} · "
                            f"severity {f.severity.lower()}")
                st.markdown(md(f.description))
                st.markdown(md("\n".join(f"1. {s_}" for s_ in ws.calculation_steps(f))))
        eb = ws.exposure_breakdown(case_id)
        if len(eb):
            st.markdown("**How the exposure is counted** — each line counts max(duplicate, quantity + max(price "
                        "mismatch, price spike)), so overlapping rules never add up twice.")
            st.dataframe(eb.rename(columns=lambda c: c.replace("_", " ").capitalize()), hide_index=True, width="stretch",
                         column_config={c: MONEY for c in ["Duplicate", "Quantity", "Price mismatch", "Price spike",
                                                           "Counted exposure", "Counted paid", "Confirmed loss"]})
        if row.related_incident_ids:
            st.markdown("**Linked cases** (share a PO line or invoice):")
            for r_ in str(row.related_incident_ids).split("|"):
                st.button(f"Open {r_}", key=f"rel_{r_}", on_click=open_case, args=(r_,))
    with tab_map:
        n, e = ws.evidence_graph(case_id)
        st.plotly_chart(charts.evidence_map(n, e), width="stretch")
        st.plotly_chart(charts.timeline(ws.timeline(case_id)), width="stretch")
    with tab_records:
        recs = ws.evidence_records(case_id)
        for name, df in recs.items():
            if len(df):
                st.markdown(f"**{name.replace('_', ' ').capitalize()}** ({len(df)})")
                st.dataframe(df, hide_index=True, width="stretch")
    with tab_next:
        st.caption("Generated from this case's evidence. Tick steps off as you go; record the outcome under Decision.")
        for i, step in enumerate(ws.playbook(case_id)):
            st.checkbox(md(step["step"]), key=f"pb_{case_id}_{i}")
            st.markdown(f"<p class='note' style='margin:-.6rem 0 .6rem 1.8rem'>{md(step['why'])}"
                        + (f" · records: {', '.join(map(str, step['records']))}" if step["records"] else "") + "</p>",
                        unsafe_allow_html=True)
    with tab_decide:
        hist = cs.history(case_id)
        with st.form(f"decide_{case_id}", clear_on_submit=True):
            c1, c2 = st.columns([2, 1])
            status = c1.selectbox("Outcome", STATUSES[1:], format_func=STATUS_TEXT.get)
            recovered = c2.number_input("Recovered amount ($)", min_value=0.0, step=100.0,
                                        help="Only with Confirmed leakage")
            note = st.text_area("Reviewer note", placeholder="What did you check, and what did you find?")
            if st.form_submit_button("Record decision", type="primary"):
                try:
                    cs.record(case_id, status, reviewer, note, recovered)
                    st.success(f"Decision recorded: {STATUS_TEXT[status]}")
                    st.rerun()
                except ValueError as err:
                    st.error(f"Decision not recorded: {err}. "
                             + ("Enter your name in the sidebar." if "reviewer" in str(err) else ""))
        if len(hist):
            st.markdown("**Decision history** (append-only)")
            st.dataframe(hist[["created_at", "reviewer", "status", "recovered_amount", "note"]], hide_index=True,
                         width="stretch")
        st.download_button("Download case report (Markdown)", ws.report_markdown(case_id, hist),
                           f"{case_id}.md", "text/markdown")

# ------------------------------------------------------------------ suppliers
elif page == "Suppliers":
    st.title("Suppliers")
    st.caption("Navigation aid: where cases and exposure concentrate. A supplier profile is not itself a finding.")
    sp = ws.suppliers.copy()
    st.plotly_chart(charts.supplier_scatter(sp), width="stretch")
    view = sp[["supplier_id", "supplier_name", "category", "country", "supplier_tier", "spend", "incidents",
               "p1_incidents", "potential_exposure", "confirmed_loss", "exposure_share_of_spend", "recurring_issues"]]
    st.dataframe(view, hide_index=True, width="stretch", height=420,
                 column_config={"spend": MONEY, "potential_exposure": MONEY, "confirmed_loss": MONEY,
                                "exposure_share_of_spend": st.column_config.NumberColumn(format="percent")})
    pick = st.selectbox("Cases for supplier", sp.supplier_id)
    sv = inc[inc.supplier_id == pick]

    def _on_sup():
        sel = st.session_state["sup_sel"]["selection"]["rows"]
        if sel:
            open_case(sv.iloc[sel[0]].incident_id)

    st.dataframe(queue_table(sv), hide_index=True, column_config=QCFG, width="stretch", key="sup_sel",
                 on_select=_on_sup, selection_mode="single-row")

# ------------------------------------------------------------------ decision log
elif page == "Decision log":
    st.title("Decision log")
    h = cs.history()
    summ = cs.summary(inc)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Decisions recorded", len(h))
    c2.metric("Confirmed leakage", summ["confirmed_leakage"])
    c3.metric("No issue (explained)", summ["no_issue"])
    c4.metric("Recovered", usd(summ["recovered_amount"]))
    if len(h):
        st.dataframe(h, hide_index=True, width="stretch")
        st.download_button("Download decision log (CSV)", h.to_csv(index=False), "decision_log.csv", "text/csv")
    else:
        st.info("No decisions yet. Open a case and record an outcome under Decision & report.")

# ------------------------------------------------------------------ evaluation
elif page == "Evaluation":
    st.title("Evaluation")
    st.warning("All numbers below come from SYNTHETIC data with planted anomalies. They show the engine recovers the "
               "planted patterns and how it degrades under simulated noise; they do not establish real-world fraud or "
               "leakage detection performance.")
    sections = [("Investigation queues: rules vs ML vs integrated", ROOT / "reports_queue" / "queue_evaluation.md",
                 [ROOT / "reports_queue" / "exposure_capture_curves.png"]),
                ("Queue order decision (pre-committed, replica dataset)", ROOT / "reports_queue" / "DECISION_RULE.md", []),
                ("Rules under realistic noise", ROOT / "reports_rules" / "noise_benchmark.md", []),
                ("Rules: confusion matrices", ROOT / "reports_rules" / "rules_benchmark.md", []),
                ("Unsupervised detectors benchmark", ROOT / "reports_ml" / "benchmark.md",
                 [ROOT / "reports_ml" / "roc_pr_curves.png", ROOT / "reports_ml" / "top1_attribution_waterfall.png"]),
                ("Detection evaluation (line level)", ROOT / "reports" / "detection_evaluation.md", [])]
    for title, md, imgs in sections:
        with st.expander(title, expanded=title.startswith("Investigation")):
            if md.exists():
                for img in imgs:
                    if img.exists():
                        st.image(str(img))
                st.markdown(md.read_text())
            else:
                st.info(f"Not generated yet: {md.relative_to(ROOT)}. See docs/PRODUCT.md for the command.")
