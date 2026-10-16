"""Plotly figures for the AI Leak Hunter app (one palette, one style)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

INK, INK2, RULE, PAPER, PANEL = "#1E2733", "#5B6676", "#D5DBE3", "#F6F7F9", "#FFFFFF"
TEAL = "#0E6E72"
# categorical slots 1-5 of the validated reference palette, fixed order, one per finding type
TYPE_COLORS = {"DUPLICATE_INVOICE": "#2a78d6", "QUANTITY_DISCREPANCY": "#eb6834", "PO_PRICE_MISMATCH": "#1baf7a",
               "HISTORICAL_PRICE_SPIKE": "#eda100", "UNUSUAL_TRANSACTION": "#e87ba4"}
TYPE_NAMES = {"DUPLICATE_INVOICE": "Duplicate invoice", "QUANTITY_DISCREPANCY": "Quantity discrepancy",
              "PO_PRICE_MISMATCH": "PO price mismatch", "HISTORICAL_PRICE_SPIKE": "Historical price spike",
              "UNUSUAL_TRANSACTION": "Statistically unusual"}
# status palette (reserved, never used for series): bands
BAND_COLORS = {"P1": "#d03b3b", "P2": "#ec835a", "P3": "#8A94A3"}


def _base(fig: go.Figure, height: int = 320, **kw) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=36, b=8), paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                      font=dict(color=INK, size=13), hoverlabel=dict(bgcolor=PANEL, font_color=INK, bordercolor=RULE),
                      **kw)
    fig.update_xaxes(gridcolor="#EEF1F4", linecolor=RULE, zeroline=False, tickfont=dict(color=INK2))
    fig.update_yaxes(gridcolor="#EEF1F4", linecolor=RULE, zeroline=False, tickfont=dict(color=INK2))
    return fig


def money_states(potential: float, paid: float, confirmed: float, pending: float) -> go.Figure:
    labels = ["Potential exposure", "Already paid out", "Confirmed loss (duplicate payments)", "Awaiting receipts (grace)"]
    vals = [potential, paid, confirmed, pending]
    notes = ["sum of de-duplicated case exposure", "share of that exposure already paid",
             "proven by payment records alone", "billed ahead of posted receipts, not yet flagged"]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker=dict(color=[TEAL, "#3F8F92", "#d03b3b", "#AEB7C2"],
                    line=dict(color=PANEL, width=2)), text=[f"${v:,.0f}" for v in vals], textposition="outside",
                    cliponaxis=False, customdata=notes,
                    hovertemplate="%{y}<br>$%{x:,.2f}<br>%{customdata}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showticklabels=False, showgrid=False, range=[0, max(vals) * 1.25 if max(vals) else 1])
    return _base(fig, 230, title=dict(text="Where the money stands", font=dict(size=15)))


def exposure_by_type(incidents: pd.DataFrame) -> go.Figure:
    g = incidents.groupby("incident_type").agg(exposure=("potential_exposure", "sum"), cases=("incident_id", "size"))
    g = g.reindex([t for t in TYPE_COLORS if t in g.index])
    fig = go.Figure(go.Bar(x=[TYPE_NAMES[t] for t in g.index], y=g.exposure, marker_color=[TYPE_COLORS[t] for t in g.index],
                           marker_line=dict(color=PANEL, width=2), customdata=g.cases,
                           text=[f"${v/1000:,.0f}k · {n} cases" for v, n in zip(g.exposure, g.cases)],
                           textposition="outside", cliponaxis=False,
                           hovertemplate="%{x}<br>$%{y:,.2f} across %{customdata} cases<extra></extra>"))
    fig.update_yaxes(tickprefix="$", tickformat="~s")
    return _base(fig, 320, title=dict(text="Potential exposure by case type", font=dict(size=15)), showlegend=False)


def exposure_by_month(incidents: pd.DataFrame, recon: pd.DataFrame) -> go.Figure:
    d = recon.set_index("invoice_line_id").invoice_date
    x = incidents.assign(month=incidents.invoice_line_ids.str.split("|").map(lambda ls: max(d[l] for l in ls)).dt.to_period("M")
                         .astype(str))
    g = x.groupby("month").agg(exposure=("potential_exposure", "sum"), cases=("incident_id", "size")).reset_index()
    fig = go.Figure(go.Bar(x=g.month, y=g.exposure, marker_color=TEAL, customdata=g.cases,
                           hovertemplate="%{x}<br>$%{y:,.2f} in %{customdata} cases<extra></extra>"))
    fig.update_yaxes(tickprefix="$", tickformat="~s")
    return _base(fig, 320, title=dict(text="Potential exposure by invoice month", font=dict(size=15)))


def risk_breakdown(inc: pd.Series) -> go.Figure:
    parts = [("Exposure", inc.pts_exposure, 35, "#2a78d6"), ("Evidence", inc.pts_evidence, 30, "#eb6834"),
             ("Money paid", inc.pts_realization, 15, "#1baf7a"), ("Statistical anomaly", inc.pts_anomaly, 20, "#eda100")]
    fig = go.Figure()
    for name, v, mx, c in parts:
        fig.add_trace(go.Bar(x=[v], y=["score"], orientation="h", name=f"{name} {v:.1f}/{mx}", marker_color=c,
                             marker_line=dict(color=PANEL, width=2),
                             hovertemplate=f"{name}: {v:.1f} of {mx} points<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(range=[0, 100], dtick=20)
    fig.update_yaxes(showticklabels=False, showgrid=False)
    return _base(fig, 150, legend=dict(orientation="h", y=-0.45, x=0, font=dict(size=12), traceorder="normal"),
                 title=dict(text=f"Risk score {inc.risk_score:.1f} / 100", font=dict(size=14)))


def evidence_map(nodes: pd.DataFrame, edges: pd.DataFrame) -> go.Figure:
    pos = {r.id: (r.layer, r.y) for r in nodes.itertuples()}
    fig = go.Figure()
    for e in edges.itertuples():
        (x0, y0), (x1, y1) = pos[e.source], pos[e.target]
        dup = e.relation == "same invoice number"
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(color="#d03b3b" if dup else "#B8C1CC", width=2.5 if dup else 1.5,
                                           dash="dot" if dup else "solid")))
    fl = nodes.flagged.astype(bool)
    for mask, color, size, name in [(~fl, "#5B6676", 18, "Supporting record"), (fl, "#d03b3b", 24, "Flagged record")]:
        n = nodes[mask]
        if n.empty:
            continue
        fig.add_trace(go.Scatter(x=n.layer, y=n.y, mode="markers+text", name=name,
                                 marker=dict(size=size, color=color, line=dict(color=PANEL, width=2)),
                                 text=n.label.str.replace("\n", "<br>"), textposition="bottom center",
                                 textfont=dict(size=11, color=INK), customdata=n.kind,
                                 hovertemplate="%{customdata}<br>%{text}<extra></extra>"))
    fig.update_xaxes(tickvals=[0, 1, 2, 3, 4], ticktext=["Supplier", "PO lines", "Amendments & receipts", "Invoice lines",
                                                         "Invoices"], range=[-0.5, 4.5], showgrid=False, side="top")
    span = max(1.0, nodes.y.abs().max() if len(nodes) else 1)
    fig.update_yaxes(showticklabels=False, showgrid=False, range=[-span - 1.0, span + 0.6])
    return _base(fig, 160 + 90 * int(nodes.groupby("layer").size().max() if len(nodes) else 1),
                 legend=dict(orientation="h", y=-0.05, x=0))


def timeline(t: pd.DataFrame) -> go.Figure:
    order = ["PO line", "Amendment (Approved)", "Amendment (Rejected)", "Goods receipt", "Invoice", "Invoice (flagged)"]
    cats = [c for c in order if c in set(t.event)] + [c for c in t.event.unique() if c not in order]
    fig = go.Figure()
    for c in cats:
        s = t[t.event == c]
        hot = c == "Invoice (flagged)"
        fig.add_trace(go.Scatter(x=s.date, y=[c] * len(s), mode="markers", name=c, showlegend=False,
                                 marker=dict(size=14 if hot else 11, color="#d03b3b" if hot else "#5B6676",
                                             symbol="diamond" if hot else "circle", line=dict(color=PANEL, width=2)),
                                 customdata=s[["record_id", "detail"]],
                                 hovertemplate="%{x|%d %b %Y}<br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>"))
    fig.update_yaxes(categoryorder="array", categoryarray=cats[::-1])
    return _base(fig, 120 + 38 * len(cats), title=dict(text="What happened when", font=dict(size=14)))


def supplier_scatter(sp: pd.DataFrame) -> go.Figure:
    s = sp[sp.spend > 0]
    fig = go.Figure(go.Scatter(x=s.spend, y=s.potential_exposure, mode="markers",
                               marker=dict(size=8 + 2 * s.incidents.clip(upper=12), color=TEAL, opacity=0.75,
                                           line=dict(color=PANEL, width=1.5)),
                               customdata=s[["supplier_id", "supplier_name", "incidents"]],
                               hovertemplate="%{customdata[0]} %{customdata[1]}<br>spend $%{x:,.0f}<br>"
                                             "exposure $%{y:,.0f}<br>%{customdata[2]} cases<extra></extra>"))
    fig.update_xaxes(title="Invoiced spend", tickprefix="$", tickformat="~s")
    fig.update_yaxes(title="Potential exposure", tickprefix="$", tickformat="~s")
    return _base(fig, 380, title=dict(text="Suppliers: spend vs exposure (marker size = cases)", font=dict(size=15)))
