#!/usr/bin/env python3
"""
generate_synthetic_data.py
==========================
Synthetic procurement dataset for AI Leak Hunter.

Produces, in ./data/ (next to this script):
  suppliers.csv          supplier master
  purchase_orders.csv    one row per PO line
  po_amendments.csv      supporting table: approved / rejected PO amendments
  goods_receipts.csv     one row per receipt line
  invoices.csv           exactly 5,000 invoice lines
  ground_truth.csv       EVALUATION ONLY - planted labels and expected outcomes
  validation_report.md   result of every validation check + distribution summary

Run:   python generate_synthetic_data.py
Needs: Python 3.9+, pandas, numpy. No network access, no paid services.
Seed:  42 (fixed). Re-running produces byte-identical files.

Pipeline stages (see main()):
  1 suppliers & catalog   2 purchase orders   3 goods receipts   4 invoices
  5 anomaly injection     6 ground truth      7 validation        8 save
  9 report

Reference detection rules (the ground-truth definitions; see README.md):
  DUPLICATE_INVOICE      >1 internal invoice document shares (supplier_id, invoice_number)
  QUANTITY_DISCREPANCY   per PO line, invoice lines ordered by (invoice_date, invoice_line_id);
                         a line is flagged when cumulative invoiced quantity exceeds the total
                         quantity received across ALL receipt lines for that PO line
  PO_PRICE_MISMATCH      invoice unit price > approved unit price, where the approved price is
                         the PO price, replaced by an Approved PRICE amendment's new price if
                         the amendment is dated on/before the invoice date
  HISTORICAL_PRICE_SPIKE invoice unit price > 1.30 x median unit price of >=5 prior invoice
                         lines (strictly earlier invoice_date) for the same supplier and item
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SEED = 42
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
START = date(2025, 10, 1)       # first possible PO date
PO_LAST = date(2026, 7, 1)      # last possible PO date
AS_OF = date(2026, 9, 30)       # data extraction date (payment status is as of this date)
CURRENCY = "USD"

N_SUPPLIERS = 80
N_INVOICE_LINES = 5000
N_ANOMALIES = 500                                 # 10% anomaly rate
N_NORMAL = N_INVOICE_LINES - N_ANOMALIES          # 4,500
# Per-type primary counts are set in N_TYPE (below the type constants).
TARGET_POS = 2300

HIST_MIN_PRIOR = 5
HIST_RATIO = 1.30           # reference rule threshold
HIST_REPAIR_RATIO = 1.25    # normal prices are kept below this (safety margin)
DUP, QTY, PRICE, HIST, NORMAL = ("DUPLICATE_INVOICE", "QUANTITY_DISCREPANCY",
                                 "PO_PRICE_MISMATCH", "HISTORICAL_PRICE_SPIKE", "NORMAL")
TYPES = [DUP, QTY, PRICE, HIST]

# Primary-label mix of the 500 anomalies. These are design choices for a varied,
# non-uniform evaluation set, NOT real-world prevalence estimates: three-way-match
# exceptions (price, quantity) are the most common planted cases, historical spikes
# are rarer because they need a 5-purchase baseline, and duplicates come in pairs.
N_TYPE = {PRICE: 170, QTY: 140, HIST: 110, DUP: 80}
N_DUP_PAIRS = N_TYPE[DUP] // 2
N_BASE_LINES = N_INVOICE_LINES - N_DUP_PAIRS     # duplicate repeats are added on top

# Sub-variants keep the same within-type shares as the previous version
# (40% undelivered balance, 10% + 10% amendment variants, 70% no contractual basis).
N_QTY_UNDELIVERED = round(0.40 * N_TYPE[QTY])          # 56: billed for goods never delivered
N_PRICE_WITH_APPROVED_AMD = round(0.10 * N_TYPE[PRICE])  # 17
N_PRICE_REJECTED_TRAP = round(0.10 * N_TYPE[PRICE])      # 17
N_HIST_NO_BASIS = round(0.70 * N_TYPE[HIST])           # 77: PO itself carries the high price

rng = np.random.default_rng(SEED)


def ri(a: int, b: int) -> int:
    """Random integer in [a, b] inclusive."""
    return int(rng.integers(a, b + 1))


def uni(a: float, b: float) -> float:
    return float(rng.uniform(a, b))


def usd(cents: int) -> str:
    return f"${cents / 100:,.2f}"


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(f"Generation invariant failed: {msg}")


# ----------------------------------------------------------------------------
# Static catalog (fictional business environment)
# ----------------------------------------------------------------------------
CATALOG = {
    "IT Hardware": [
        ("Business laptop 14in", 1150), ("27in LED monitor", 240), ("USB-C docking station", 185),
        ("Wireless keyboard and mouse set", 45), ("24-port managed network switch", 620),
        ("1TB NVMe SSD", 95), ("1U rack server", 4800), ("Enterprise Wi-Fi access point", 380),
        ("Mono laser printer", 310), ("UPS 1500VA", 520)],
    "Office Supplies": [
        ("A4 copy paper (box of 5 reams)", 28), ("Ballpoint pens (box of 50)", 12.5),
        ("Toner cartridge black", 88), ("Lever arch files (pack of 10)", 19),
        ("Sticky notes (pack of 12)", 9.8), ("Whiteboard markers (pack of 8)", 11),
        ("Ergonomic office chair", 340), ("Desk organizer set", 24),
        ("Shredder bags (roll of 50)", 21), ("Notebook A5 (pack of 10)", 17)],
    "MRO & Tools": [
        ("Cordless drill 18V", 165), ("Hex bolt M10 (box of 100)", 22), ("Ball bearing 6205", 7.4),
        ("Hydraulic oil ISO 46 (20L)", 96), ("V-belt B-section", 14.5), ("Pipe wrench 18in", 38),
        ("Industrial grease cartridge", 8.9), ("Air filter element", 42),
        ("Welding rods (5kg)", 31), ("Safety relief valve 1in", 215)],
    "Packaging": [
        ("Corrugated box 40x30x30 (bundle of 25)", 31), ("Stretch wrap film roll", 18.5),
        ("Wooden pallet 1200x1000", 14), ("Packing tape (case of 36)", 54),
        ("Bubble wrap roll 100m", 39), ("Pallet strapping kit", 125),
        ("Kraft void-fill paper roll", 47), ("Shipping labels (roll of 1000)", 16),
        ("Polybags 30x40 (pack of 500)", 29), ("Edge protectors (bundle of 100)", 44)],
    "Lab & Chemicals": [
        ("Nitrile lab gloves (box of 100)", 11.5), ("Isopropyl alcohol 99% (5L)", 38),
        ("Pipette tips 200uL (rack of 960)", 64), ("Sodium hydroxide pellets (1kg)", 27),
        ("Borosilicate beaker set", 58), ("Distilled water (20L)", 19),
        ("Centrifuge tubes 50mL (pack of 500)", 92), ("pH buffer solution kit", 46),
        ("Filter paper (box of 100)", 15), ("Calibration weight set", 410)],
    "Electrical Components": [
        ("Copper cable 2.5mm2 (100m)", 86), ("Circuit breaker 32A", 23), ("LED panel light 60x60", 41),
        ("Cable tray 3m section", 37), ("Industrial contactor 25A", 58),
        ("Distribution board 12-way", 145), ("Variable frequency drive 5.5kW", 690),
        ("Junction box IP65", 12), ("Terminal blocks (pack of 50)", 26), ("Digital multimeter", 120)],
    "Safety & PPE": [
        ("Safety helmet", 14), ("High-visibility vest", 6.5), ("Steel-toe safety boots", 62),
        ("Safety goggles", 7.8), ("Ear defenders", 19), ("FFP2 respirator masks (box of 20)", 24),
        ("Cut-resistant gloves (pair)", 9.5), ("Fall arrest harness", 145),
        ("Workplace first aid kit", 48), ("Fire extinguisher 6kg", 72)],
    "Facilities Consumables": [
        ("Hand soap refill 5L", 16), ("Paper hand towels (case)", 34),
        ("Floor cleaner concentrate 5L", 22), ("Heavy-duty trash bags (roll of 100)", 18),
        ("Microfiber cloths (pack of 50)", 27), ("HVAC filter MERV 13", 31),
        ("Toilet tissue (case of 48)", 39), ("Disinfectant spray (case of 12)", 44),
        ("Mop head replacement", 8.5), ("LED tube 4ft (pack of 10)", 56)],
}
FIRST_WORDS = ["Ashcombe", "Birchfield", "Bluestone", "Brightwater", "Cedarline", "Coldspring",
               "Copperfield", "Driftwood", "Dunmore", "Eastbrook", "Elmstead", "Fairhaven",
               "Foxglove", "Glenmore", "Greystone", "Harborview", "Hollybrook", "Ironbridge",
               "Ivydale", "Juniper", "Kestrel", "Kingsmere", "Larchmont", "Lowbridge",
               "Maplecrest", "Millbrook", "Newhaven", "Northbay", "Oakhurst", "Pinewood",
               "Quayside", "Ravenscroft", "Silverbirch", "Stonegate", "Thornbury", "Upland",
               "Valemont", "Westhollow", "Wrenfield", "Yarrowdale"]
CAT_WORDS = {
    "IT Hardware": ["Tech Supply", "Digital Systems", "Computing", "IT Distribution"],
    "Office Supplies": ["Office Supply", "Stationers", "Workplace Goods", "Office Essentials"],
    "MRO & Tools": ["Industrial Supply", "Tool & Fastener", "Maintenance Supply", "Engineering Stores"],
    "Packaging": ["Packaging", "Pack Solutions", "Containers", "Wrap & Pack"],
    "Lab & Chemicals": ["Scientific", "Lab Supply", "Chemicals", "Laboratory Products"],
    "Electrical Components": ["Electrical", "Electric Supply", "Power Components", "Cable & Control"],
    "Safety & PPE": ["Safety Supply", "Protective Equipment", "Safety Products", "Workwear & PPE"],
    "Facilities Consumables": ["Facility Supply", "Janitorial", "Facilities Products", "Hygiene Supply"],
}
COUNTRIES = ["United States", "Canada", "Mexico", "Germany", "United Kingdom", "Netherlands",
             "Poland", "Spain", "Italy", "India", "China", "Japan", "South Korea", "Turkey"]
COUNTRY_P = [.40, .07, .06, .08, .07, .04, .04, .03, .03, .06, .05, .03, .02, .02]
LEGAL = {"United States": ["Inc.", "LLC", "Corp."], "Canada": ["Inc.", "Ltd."],
         "Mexico": ["S.A. de C.V."], "Germany": ["GmbH"], "United Kingdom": ["Ltd."],
         "Netherlands": ["B.V."], "Poland": ["Sp. z o.o."], "Spain": ["S.L."], "Italy": ["S.r.l."],
         "India": ["Pvt. Ltd."], "China": ["Co., Ltd."], "Japan": ["K.K."],
         "South Korea": ["Co., Ltd."], "Turkey": ["A.S."]}
TIERS, TIER_P = ["Small", "Medium", "Large"], [.45, .35, .20]
TIER_W = {"Small": 1.0, "Medium": 2.5, "Large": 5.0}      # purchasing frequency
TIER_Q = {"Small": 1.0, "Medium": 1.5, "Large": 2.5}      # order-size multiplier


# ----------------------------------------------------------------------------
# Internal state
# ----------------------------------------------------------------------------
class State:
    def __init__(self):
        self.items, self.suppliers = [], []
        self.pos, self.plines, self.amds = {}, {}, {}
        self.receipts, self.docs, self.ilines = {}, {}, {}
        self.plan = {}          # iline key -> planted anomaly metadata
        self.forced_status = {}  # optional override: doc key -> payment status (unused by default)
        self.stats = Counter()
        self._ctr = Counter()

    def nk(self, kind: str) -> int:
        self._ctr[kind] += 1
        return self._ctr[kind]

    def ddate(self, line) -> date:
        return self.docs[line["doc_key"]]["date"]

    def dsort(self, line):
        d = self.docs[line["doc_key"]]
        return (d["date"], d["tb"], d["key"])


def approved_price_c(S: State, pl: dict, dt: date) -> int:
    """Approved unit price (cents) applicable on date dt."""
    a = S.amds.get(pl["amd_key"]) if pl["amd_key"] is not None else None
    if a and a["type"] == "PRICE" and a["status"] == "Approved" and dt >= a["date"]:
        return a["new_price_c"]
    return pl["price_c"]


def lines_by_pl(S: State):
    d = defaultdict(list)
    for l in S.ilines.values():
        d[l["pl_key"]].append(l)
    for v in d.values():
        v.sort(key=S.dsort)
    return d


def new_doc(S, po_key, supplier_id, dt, items, copy_of=None):
    dk = S.nk("doc")
    S.docs[dk] = dict(key=dk, po_key=po_key, supplier_id=supplier_id, date=dt,
                      copy_of=copy_of, tb=float(rng.random()), lines=[])
    for pl, q in items:
        lk = S.nk("iline")
        S.ilines[lk] = dict(key=lk, doc_key=dk, pl_key=pl["key"], qty=int(q),
                            price_c=approved_price_c(S, pl, dt))
        S.docs[dk]["lines"].append(lk)
    return dk


# ----------------------------------------------------------------------------
# Stage 1 - suppliers and product catalog
# ----------------------------------------------------------------------------
def stage1_suppliers_and_catalog(S: State):
    n = 0
    for cat, lst in CATALOG.items():
        for name, price in lst:
            n += 1
            S.items.append(dict(item_id=f"ITM-{n:04d}", item_description=name,
                                category=cat, base_price=float(price)))
    by_cat = defaultdict(list)
    for it in S.items:
        by_cat[it["category"]].append(it)

    cats = list(CATALOG)
    sup_cats = [c for c in cats for _ in range(N_SUPPLIERS // len(cats))]
    rng.shuffle(sup_cats)
    names = set()
    for i in range(N_SUPPLIERS):
        cat = sup_cats[i]
        country = str(rng.choice(COUNTRIES, p=COUNTRY_P))
        tier = str(rng.choice(TIERS, p=TIER_P))
        while True:
            first = str(rng.choice(FIRST_WORDS))
            nm = f"{first} {rng.choice(CAT_WORDS[cat])} {rng.choice(LEGAL[country])}"
            if nm not in names:
                break
        names.add(nm)
        if tier == "Small":
            k = ri(3, 4)
        elif tier == "Medium":
            k = ri(4, 6)
        else:
            k = ri(6, 8)
        sitems = []
        for j in rng.choice(len(by_cat[cat]), size=k, replace=False):
            sitems.append(dict(item=by_cat[cat][int(j)],
                               factor=float(np.clip(rng.lognormal(0, 0.08), 0.8, 1.25)),
                               drift=uni(0.0, 0.04)))
        fmt = int(rng.integers(0, 5))
        words = nm.split()
        S.suppliers.append(dict(
            supplier_id=f"SUP-{i + 1:04d}", supplier_name=nm, category=cat, country=country,
            supplier_tier=tier, variable=bool(rng.random() < 0.15), items=sitems,
            inv_fmt=fmt, inv_seq=ri(1000, 60000), pfx=(words[0][0] + words[1][0]).upper()))


# ----------------------------------------------------------------------------
# Stage 2 - purchase orders (+ amendments)
# ----------------------------------------------------------------------------
def stage2_purchase_orders(S: State):
    span = (PO_LAST - START).days
    total_w = sum(TIER_W[s["supplier_tier"]] for s in S.suppliers)
    for s in S.suppliers:
        n_po = max(5, int(round(TARGET_POS * TIER_W[s["supplier_tier"]] / total_w)))
        sd, cap = (0.05, 0.10) if s["variable"] else (0.015, 0.04)
        for off in np.sort(rng.integers(0, span + 1, n_po)):
            off = int(off)
            pk = S.nk("po")
            po = dict(key=pk, supplier_id=s["supplier_id"], po_date=START + timedelta(days=off),
                      lead=ri(3, 21), tb=float(rng.random()), lines=[])
            S.pos[pk] = po
            n_lines = min(int(rng.choice([1, 2, 3, 4], p=[.40, .30, .20, .10])), len(s["items"]))
            for ln, j in enumerate(rng.choice(len(s["items"]), size=n_lines, replace=False), start=1):
                si = s["items"][int(j)]
                it = si["item"]
                noise = float(np.clip(rng.normal(0, sd), -cap, cap))
                price = it["base_price"] * si["factor"] * (1 + si["drift"] * off / 365.0) * (1 + noise)
                price_c = max(1, int(round(price * 100)))
                typical = float(np.clip(2500.0 / it["base_price"], 1, 300)) * TIER_Q[s["supplier_tier"]]
                q = max(1, int(round(rng.lognormal(np.log(typical), 0.45))))
                if rng.random() < 0.03:          # occasional large but legitimate order
                    q *= ri(4, 8)
                if q > 20:
                    q = int(round(q / 5.0) * 5)
                lk = S.nk("pl")
                pl = dict(key=lk, po_key=pk, line_no=ln, supplier_id=s["supplier_id"],
                          item_id=it["item_id"], item_description=it["item_description"],
                          qty_ordered=q, final_qty=q, price_c=price_c, amd_key=None)
                S.plines[lk] = pl
                po["lines"].append(lk)
                u = rng.random()
                if u < 0.10:
                    ak = S.nk("amd")
                    adate = po["po_date"] + timedelta(days=ri(1, po["lead"]))
                    a = dict(key=ak, pl_key=lk, date=adate, tb=float(rng.random()),
                             old_price_c=None, new_price_c=None, old_qty=None, new_qty=None,
                             decided_by=str(rng.choice(["Category Manager", "Procurement Director",
                                                        "Finance Controller"])))
                    if u < 0.06:
                        a.update(type="PRICE", status="Approved", old_price_c=price_c,
                                 new_price_c=max(price_c + 1, int(round(price_c * (1 + uni(.02, .06))))))
                    elif u < 0.08:
                        nq = q + max(1, int(round(q * uni(.10, .50))))
                        a.update(type="QUANTITY", status="Approved", old_qty=q, new_qty=nq)
                        pl["final_qty"] = nq
                    else:
                        a.update(type="PRICE", status="Rejected", old_price_c=price_c,
                                 new_price_c=max(price_c + 1, int(round(price_c * (1 + uni(.04, .12))))))
                    S.amds[ak] = a
                    pl["amd_key"] = ak


# ----------------------------------------------------------------------------
# Stage 3 - goods receipts (single, split and short shipments)
# ----------------------------------------------------------------------------
def stage3_goods_receipts(S: State):
    for po in S.pos.values():
        n_ship = int(rng.choice([1, 2, 3], p=[.60, .28, .12]))
        d = [po["po_date"] + timedelta(days=po["lead"])]
        for _ in range(n_ship - 1):
            d.append(d[-1] + timedelta(days=ri(7, 25)))
        all_parts = {}
        for lk in po["lines"]:
            pl = S.plines[lk]
            Q = pl["final_qty"]
            parts = [0] * n_ship
            if n_ship == 1:
                parts[0] = Q
            elif Q >= n_ship and rng.random() < 0.5:
                extra = rng.multinomial(Q - n_ship, rng.dirichlet(np.ones(n_ship)))
                parts = [1 + int(x) for x in extra]
            else:
                parts[int(rng.integers(0, n_ship))] = Q
            if Q >= 2 and rng.random() < 0.06:   # supplier short-ships; billed as received
                short = ri(1, max(1, int(0.2 * Q)))
                for idx in range(n_ship - 1, -1, -1):
                    take = min(short, parts[idx])
                    parts[idx] -= take
                    short -= take
            all_parts[lk] = parts
        # keep only shipments that actually delivered something, so the PO's first
        # shipment date is always a real delivery (no invoice precedes every delivery)
        used = [i for i in range(n_ship) if any(p[i] > 0 for p in all_parts.values())]
        po["ship_dates"] = [d[i] for i in used]
        for lk, parts in all_parts.items():
            pl = S.plines[lk]
            pl["parts"] = [parts[i] for i in used]
            pl["received"] = sum(parts)
            for idx, qv in enumerate(pl["parts"]):
                if qv > 0:
                    rk = S.nk("rcpt")
                    S.receipts[rk] = dict(key=rk, pl_key=lk, date=po["ship_dates"][idx], qty=int(qv),
                                          tb=float(rng.random()))


# ----------------------------------------------------------------------------
# Stage 4 - ordinary invoices, trimming, price normalisation
# ----------------------------------------------------------------------------
def stage4_invoices(S: State):
    for po in S.pos.values():
        d = po["ship_dates"]
        lines = [S.plines[k] for k in po["lines"]]
        if len(d) == 1:
            mode = "single"
        else:
            mode = str(rng.choice(["per_shipment", "upfront", "after_complete"], p=[.45, .25, .30]))
        po["billing_mode"] = mode
        if mode == "per_shipment":
            for idx, sd in enumerate(d):
                bl = [(pl, pl["parts"][idx]) for pl in lines if pl["parts"][idx] > 0]
                if bl:
                    new_doc(S, po["key"], po["supplier_id"], sd + timedelta(days=ri(0, 10)), bl)
        elif mode == "upfront":   # billed in full at first delivery; rest arrives later
            new_doc(S, po["key"], po["supplier_id"], d[0] + timedelta(days=ri(0, 7)),
                    [(pl, pl["received"]) for pl in lines])
        else:
            new_doc(S, po["key"], po["supplier_id"], d[-1] + timedelta(days=ri(0, 10)),
                    [(pl, pl["received"]) for pl in lines])
    S.stats["base_lines_generated"] = len(S.ilines)


def _drop_po(S: State, pk: int):
    po = S.pos.pop(pk)
    for lk in po["lines"]:
        pl = S.plines.pop(lk)
        if pl["amd_key"] is not None:
            S.amds.pop(pl["amd_key"], None)
    for rk in [k for k, r in S.receipts.items() if r["pl_key"] in po["lines"]]:
        del S.receipts[rk]
    for dk in [k for k, d in S.docs.items() if d["po_key"] == pk]:
        for lk in S.docs[dk]["lines"]:
            del S.ilines[lk]
        del S.docs[dk]


def stage4b_trim(S: State):
    """Drop whole POs at random, then the most recent invoice lines (received but not yet
    invoiced at extraction) so exactly N_BASE_LINES ordinary invoice lines remain."""
    per_po = Counter(S.docs[l["doc_key"]]["po_key"] for l in S.ilines.values())
    total = len(S.ilines)
    require(total >= N_BASE_LINES, f"only {total} base lines; increase TARGET_POS")
    for pk in rng.permutation(sorted(S.pos)):
        pk = int(pk)
        if total == N_BASE_LINES:
            break
        if total - per_po.get(pk, 0) >= N_BASE_LINES and total - per_po.get(pk, 0) > N_BASE_LINES - 3:
            total -= per_po.get(pk, 0)
            _drop_po(S, pk)
    latest = sorted(S.ilines.values(), key=S.dsort, reverse=True)
    i = 0
    while len(S.ilines) > N_BASE_LINES:
        l = latest[i]
        i += 1
        doc = S.docs[l["doc_key"]]
        doc["lines"].remove(l["key"])
        del S.ilines[l["key"]]
        if not doc["lines"]:
            del S.docs[doc["key"]]
    require(len(S.ilines) == N_BASE_LINES, "trim failed")


def hist_stats(S: State):
    groups = defaultdict(list)
    for l in S.ilines.values():
        pl = S.plines[l["pl_key"]]
        groups[(pl["supplier_id"], pl["item_id"])].append(l)
    out = {}
    for g in groups.values():
        dates = [S.ddate(l) for l in g]
        for l, dt in zip(g, dates):
            prior = [m["price_c"] for m, dm in zip(g, dates) if dm < dt]
            if len(prior) >= HIST_MIN_PRIOR:
                med = float(np.median(prior))
                out[l["key"]] = (len(prior), med, l["price_c"] / med)
            else:
                out[l["key"]] = (len(prior), None, None)
    return out, groups


def stage4c_normalise_prices(S: State):
    """Natural price variation must never trip the historical-spike rule on normal lines.
    Any ordinary PO line whose price exceeds HIST_REPAIR_RATIO x its prior median is re-priced
    (PO price, amendment prices and invoice prices together) back near the baseline."""
    repairs = 0
    for _ in range(50):
        st, _ = hist_stats(S)
        viol = [(k, v) for k, v in st.items() if v[2] is not None and v[2] > HIST_REPAIR_RATIO]
        if not viol:
            break
        lbp = lines_by_pl(S)
        done = set()
        for k, (_, med, _) in viol:
            pl = S.plines[S.ilines[k]["pl_key"]]
            if pl["key"] in done:
                continue
            done.add(pl["key"])
            f = med / S.ilines[k]["price_c"] * uni(0.98, 1.02)
            pl["price_c"] = max(1, int(round(pl["price_c"] * f)))
            if pl["amd_key"] is not None:
                a = S.amds[pl["amd_key"]]
                if a["type"] == "PRICE":
                    a["old_price_c"] = pl["price_c"]
                    a["new_price_c"] = max(pl["price_c"] + 1, int(round(a["new_price_c"] * f)))
            for l in lbp[pl["key"]]:
                l["price_c"] = approved_price_c(S, pl, S.ddate(l))
            repairs += 1
    st, _ = hist_stats(S)
    require(all(v[2] is None or v[2] <= HIST_REPAIR_RATIO for v in st.values()), "price normalisation")
    S.stats["normal_price_repairs"] = repairs


# ----------------------------------------------------------------------------
# Stage 5 - controlled anomaly injection
# ----------------------------------------------------------------------------
def _tier(probs, ranges, names):
    t = int(rng.choice(len(probs), p=probs))
    return uni(*ranges[t]), names[t]


def stage5_inject_anomalies(S: State):
    lbp = lines_by_pl(S)
    st, _ = hist_stats(S)
    billed = {pk: sum(l["qty"] for l in v) for pk, v in lbp.items()}
    used = set()
    all_lines = sorted(S.ilines.values(), key=lambda l: l["key"])

    def fully_billed(pk):
        return billed.get(pk, 0) == S.plines[pk]["received"]

    def is_last(l):
        return lbp[l["pl_key"]][-1]["key"] == l["key"]

    def pick_distinct(cands, n, cap_pair=None):
        out, per_pair = [], Counter()
        for i in rng.permutation(len(cands)):
            l = cands[int(i)]
            pl = S.plines[l["pl_key"]]
            if pl["key"] in used:
                continue
            if cap_pair is not None:
                pair = (pl["supplier_id"], pl["item_id"])
                if per_pair[pair] >= cap_pair:
                    continue
                per_pair[pair] += 1
            used.add(pl["key"])
            out.append(l)
            if len(out) == n:
                break
        require(len(out) == n, f"not enough candidates ({len(out)}/{n})")
        return out

    # 5.1 reserve historical-spike lines (most constrained: need >= 5 prior lines)
    cands = [l for l in all_lines if len(lbp[l["pl_key"]]) == 1
             and S.plines[l["pl_key"]]["amd_key"] is None and st[l["key"]][0] >= HIST_MIN_PRIOR]
    hist_sel = pick_distinct(cands, N_TYPE[HIST], cap_pair=2)

    # 5.2 duplicate invoices: re-enter a single-line document under a new internal ID
    cands = [l for l in all_lines if len(S.docs[l["doc_key"]]["lines"]) == 1 and is_last(l)
             and fully_billed(l["pl_key"]) and S.ddate(l) <= AS_OF - timedelta(days=1)]
    dup_sel = pick_distinct(cands, N_DUP_PAIRS)
    for g, l in enumerate(sorted(dup_sel, key=S.dsort), start=1):
        od = S.docs[l["doc_key"]]
        pl = S.plines[l["pl_key"]]
        cdate = min(od["date"] + timedelta(days=ri(1, 30)), AS_OF)
        ck = new_doc(S, od["po_key"], od["supplier_id"], cdate, [(pl, l["qty"])], copy_of=od["key"])
        cl = S.ilines[S.docs[ck]["lines"][0]]
        cl["price_c"] = l["price_c"]
        amount = l["qty"] * l["price_c"]
        S.plan[l["key"]] = dict(primary=DUP, tags=["DUP_FIRST_ENTRY"], group=g, partner=cl["key"],
                                exposure_c=0, amount_c=amount)
        S.plan[cl["key"]] = dict(primary=DUP, tags=["DUP_REPEAT_ENTRY"], group=g, partner=l["key"],
                                 exposure_c=amount, amount_c=amount)

    # 5.3 quantity discrepancies
    def excess_for(R):
        frac, size = _tier([.4, .4, .2], [(.02, .10), (.10, .35), (.35, 1.0)],
                           ["EXCESS_SMALL", "EXCESS_MEDIUM", "EXCESS_LARGE"])
        return max(1, int(round(R * frac))), size

    cands = [l for l in all_lines if is_last(l) and fully_billed(l["pl_key"])
             and len(lbp[l["pl_key"]]) == 1 and S.plines[l["pl_key"]]["received"] >= 2]
    for l in pick_distinct(cands, N_QTY_UNDELIVERED):   # billed for goods never delivered
        pl = S.plines[l["pl_key"]]
        R = pl["received"]
        e, size = excess_for(R)
        e = min(e, R - 1)
        rcpts = sorted([r for r in S.receipts.values() if r["pl_key"] == pl["key"]],
                       key=lambda r: (r["date"], r["tb"], r["key"]), reverse=True)
        rem = e
        for r in rcpts:
            if rem == 0:
                break
            take = min(rem, r["qty"])
            r["qty"] -= take
            rem -= take
            if r["qty"] == 0:
                del S.receipts[r["key"]]
        pl["received"] -= e
        appr = approved_price_c(S, pl, S.ddate(l))
        S.plan[l["key"]] = dict(primary=QTY, tags=["UNDELIVERED_BALANCE_BILLED", size], excess=e,
                                appr_c=appr, exposure_c=e * appr)
    cands = [l for l in all_lines if is_last(l) and fully_billed(l["pl_key"])]
    for l in pick_distinct(cands, N_TYPE[QTY] - N_QTY_UNDELIVERED):  # invoice qty inflated
        pl = S.plines[l["pl_key"]]
        e, size = excess_for(pl["received"])
        l["qty"] += e
        appr = approved_price_c(S, pl, S.ddate(l))
        S.plan[l["key"]] = dict(primary=QTY, tags=["INVOICE_QTY_INFLATED", size], excess=e,
                                appr_c=appr, exposure_c=e * appr)

    # 5.4 PO price mismatches
    def has_amd(l, typ, status):
        a = S.amds.get(S.plines[l["pl_key"]]["amd_key"])
        return a is not None and a["type"] == typ and a["status"] == status

    sel = [(l, "ABOVE_APPROVED_AMENDMENT") for l in pick_distinct(
        [l for l in all_lines if has_amd(l, "PRICE", "Approved")], N_PRICE_WITH_APPROVED_AMD)]
    sel += [(l, "REJECTED_AMENDMENT_PRICE_BILLED") for l in pick_distinct(
        [l for l in all_lines if has_amd(l, "PRICE", "Rejected")], N_PRICE_REJECTED_TRAP)]
    sel += [(l, "NO_AMENDMENT") for l in pick_distinct(
        [l for l in all_lines if S.plines[l["pl_key"]]["amd_key"] is None],
        N_TYPE[PRICE] - len(sel))]
    for l, variant in sel:
        pl = S.plines[l["pl_key"]]
        appr = approved_price_c(S, pl, S.ddate(l))
        if variant == "REJECTED_AMENDMENT_PRICE_BILLED":
            new = S.amds[pl["amd_key"]]["new_price_c"]
        else:
            m, _ = _tier([.4, .4, .2], [(.02, .08), (.08, .20), (.20, .60)], ["", "", ""])
            new = max(appr + 1, int(round(appr * (1 + m))))
        r = new / appr - 1
        size = "OVERCHARGE_MODEST" if r < .08 else ("OVERCHARGE_MODERATE" if r < .20 else "OVERCHARGE_SUBSTANTIAL")
        require(new > appr, "price anomaly must exceed approved price")
        l["price_c"] = new
        S.plan[l["key"]] = dict(primary=PRICE, tags=[variant, size], appr_c=appr, new_c=new,
                                exposure_c=(new - appr) * l["qty"])

    # 5.5 historical spikes, planted chronologically so each baseline is final when used
    _, groups = hist_stats(S)
    variants = ["NO_CONTRACT_BASIS"] * N_HIST_NO_BASIS + ["ALSO_ABOVE_PO_PRICE"] * (N_TYPE[HIST] - N_HIST_NO_BASIS)
    variants = [variants[int(i)] for i in rng.permutation(len(variants))]
    for l, variant in zip(sorted(hist_sel, key=S.dsort), variants):
        pl = S.plines[l["pl_key"]]
        dt = S.ddate(l)
        prior = sorted([m for m in groups[(pl["supplier_id"], pl["item_id"])] if S.ddate(m) < dt],
                       key=S.dsort)
        require(len(prior) >= HIST_MIN_PRIOR, "historical baseline too short")
        med = float(np.median([m["price_c"] for m in prior]))
        r, size = _tier([.4, .4, .2], [(1.40, 1.70), (1.70, 2.20), (2.20, 3.00)],
                        ["SPIKE_MODERATE", "SPIKE_HIGH", "SPIKE_EXTREME"])
        new = int(math.ceil(med * r))
        if variant == "NO_CONTRACT_BASIS":     # the PO itself was approved at the high price
            pl["price_c"] = new
            l["price_c"] = new
            appr, expo = new, 0
        else:                                  # invoice exceeds both PO price and history
            appr = approved_price_c(S, pl, dt)
            require(new > appr, "spike must exceed PO price")
            l["price_c"] = new
            expo = (new - appr) * l["qty"]
        S.plan[l["key"]] = dict(primary=HIST, tags=[variant, size], appr_c=appr, new_c=new,
                                exposure_c=expo, n_prior=len(prior), median_c=med,
                                prior_keys=[m["key"] for m in prior])


def stage5b_payments(S: State):
    for d in sorted(S.docs.values(), key=lambda d: (d["date"], d["tb"], d["key"])):
        if d["key"] in S.forced_status:
            stt = S.forced_status[d["key"]]
        else:
            age = (AS_OF - d["date"]).days
            p = [.82, .08, .10] if age > 75 else ([.50, .20, .30] if age > 30 else [.15, .15, .70])
            stt = str(rng.choice(["Paid", "Partially Paid", "Unpaid"], p=p))
        d["status"], d["frac"] = stt, uni(.25, .75)


def format_invoice_number(s, dt):
    f, q = s["inv_fmt"], s["inv_seq"]
    if f == 0:
        return f"INV-{dt.year}-{q:06d}"
    if f == 1:
        return f"{s['pfx']}{q:07d}"
    if f == 2:
        return f"SI/{dt.year % 100:02d}/{q:06d}"
    if f == 3:
        return f"{10_000_000 + q:08d}"
    return f"{dt.year}{dt.month:02d}-{q:06d}"


def stage5c_invoice_numbers(S: State):
    sups = {s["supplier_id"]: s for s in S.suppliers}
    docs = sorted(S.docs.values(), key=lambda d: (d["date"], d["tb"], d["key"]))
    for d in docs:
        if d["copy_of"] is None:
            s = sups[d["supplier_id"]]
            s["inv_seq"] += ri(1, 40)   # supplier also bills other customers
            d["invoice_number"] = format_invoice_number(s, d["date"])
    for d in docs:
        if d["copy_of"] is not None:
            d["invoice_number"] = S.docs[d["copy_of"]]["invoice_number"]


# ----------------------------------------------------------------------------
# Stage 6 - tables, reference rules, ground truth
# ----------------------------------------------------------------------------
def paid_c(doc, amount_c):
    if doc["status"] == "Paid":
        return amount_c
    if doc["status"] == "Unpaid":
        return 0
    return int(min(amount_c - 1, max(1, round(amount_c * doc["frac"]))))


def stage6_build_tables(S: State):
    ids = dict(po={}, pl={}, amd={}, rc={}, doc={}, il={})
    sup = pd.DataFrame([{k: s[k] for k in ["supplier_id", "supplier_name", "category", "country",
                                            "supplier_tier"]} for s in S.suppliers])
    for n, a in enumerate(sorted(S.amds.values(), key=lambda a: (a["date"], a["tb"], a["key"])), 1):
        ids["amd"][a["key"]] = f"AMD-{n:05d}"
    po_rows = []
    for n, p in enumerate(sorted(S.pos.values(), key=lambda p: (p["po_date"], p["tb"], p["key"])), 1):
        ids["po"][p["key"]] = pid = f"PO-{n:06d}"
        for lk in p["lines"]:
            pl = S.plines[lk]
            ids["pl"][lk] = plid = f"{pid}-{pl['line_no']:02d}"
            a = S.amds.get(pl["amd_key"]) if pl["amd_key"] is not None else None
            po_rows.append(dict(po_id=pid, po_line_id=plid, supplier_id=p["supplier_id"],
                                po_date=p["po_date"], item_id=pl["item_id"],
                                item_description=pl["item_description"],
                                quantity_ordered=pl["qty_ordered"], unit_price=pl["price_c"] / 100,
                                currency=CURRENCY,
                                approved_amendment_id=ids["amd"][a["key"]] if a and a["status"] == "Approved" else ""))
    po = pd.DataFrame(po_rows)
    amd_rows = []
    for a in sorted(S.amds.values(), key=lambda a: ids["amd"][a["key"]]):
        pl = S.plines[a["pl_key"]]
        amd_rows.append(dict(amendment_id=ids["amd"][a["key"]], po_id=ids["po"][pl["po_key"]],
                             po_line_id=ids["pl"][pl["key"]], amendment_date=a["date"],
                             amendment_type=a["type"], status=a["status"],
                             previous_unit_price=a["old_price_c"] / 100 if a["old_price_c"] else np.nan,
                             new_unit_price=a["new_price_c"] / 100 if a["new_price_c"] else np.nan,
                             previous_quantity=a["old_qty"], new_quantity=a["new_qty"],
                             decided_by=a["decided_by"]))
    amd = pd.DataFrame(amd_rows)
    for c in ["previous_quantity", "new_quantity"]:
        amd[c] = amd[c].astype("Int64")
    gr_rows = []
    for n, r in enumerate(sorted(S.receipts.values(), key=lambda r: (r["date"], r["tb"], r["key"])), 1):
        ids["rc"][r["key"]] = rid = f"GR-{n:06d}"
        pl = S.plines[r["pl_key"]]
        gr_rows.append(dict(receipt_id=rid, po_id=ids["po"][pl["po_key"]], po_line_id=ids["pl"][pl["key"]],
                            receipt_date=r["date"], item_id=pl["item_id"], quantity_received=r["qty"]))
    gr = pd.DataFrame(gr_rows)
    inv_rows = []
    for n, d in enumerate(sorted(S.docs.values(), key=lambda d: (d["date"], d["tb"], d["key"])), 1):
        ids["doc"][d["key"]] = iid = f"INV-{n:06d}"
        lines = sorted((S.ilines[k] for k in d["lines"]), key=lambda l: S.plines[l["pl_key"]]["line_no"])
        for j, l in enumerate(lines, 1):
            ids["il"][l["key"]] = ilid = f"{iid}-{j:02d}"
            pl = S.plines[l["pl_key"]]
            amount = l["qty"] * l["price_c"]
            inv_rows.append(dict(invoice_id=iid, invoice_number=d["invoice_number"], invoice_line_id=ilid,
                                 supplier_id=d["supplier_id"], po_id=ids["po"][pl["po_key"]],
                                 po_line_id=ids["pl"][pl["key"]], invoice_date=d["date"],
                                 item_id=pl["item_id"], quantity_invoiced=l["qty"],
                                 unit_price=l["price_c"] / 100, currency=CURRENCY,
                                 line_amount=amount / 100, payment_status=d["status"],
                                 amount_paid=paid_c(d, amount) / 100))
    inv = pd.DataFrame(inv_rows)
    return dict(suppliers=sup, purchase_orders=po, po_amendments=amd, goods_receipts=gr, invoices=inv), ids


def cents(s: pd.Series) -> pd.Series:
    return (s.astype(float) * 100).round().astype("int64")


def reference_rules(po, gr, inv, amd) -> pd.DataFrame:
    """The four reference detection rules, computed only from the procurement tables."""
    df = inv.copy()
    df["_pos"] = np.arange(len(df))
    df["price_c"] = cents(df.unit_price)
    pox = po.set_index("po_line_id")
    df["po_price_c"] = cents(df.po_line_id.map(pox.unit_price))
    df["po_amendment_id"] = df.po_line_id.map(pox.approved_amendment_id)
    ap = amd[(amd.status == "Approved") & (amd.amendment_type == "PRICE")].set_index("amendment_id")
    new_p = df.po_amendment_id.map(ap.new_unit_price)
    a_dt = df.po_amendment_id.map(ap.amendment_date)
    df["amendment_applied"] = new_p.notna() & (a_dt <= df.invoice_date)
    df["approved_price_c"] = np.where(df.amendment_applied, cents(new_p.fillna(0)), df.po_price_c)
    df["price_excess_c"] = (df.price_c - df.approved_price_c).clip(lower=0)
    df["flag_price"] = df.price_excess_c > 0

    df["received_total"] = df.po_line_id.map(gr.groupby("po_line_id").quantity_received.sum()).fillna(0).astype(int)
    df = df.sort_values(["invoice_date", "invoice_line_id"], kind="mergesort")
    df["cum_invoiced"] = df.groupby("po_line_id").quantity_invoiced.cumsum()
    df["excess_qty"] = np.minimum(df.quantity_invoiced, (df.cum_invoiced - df.received_total).clip(lower=0))
    df["flag_qty"] = df.excess_qty > 0
    rec = defaultdict(list)
    for p, d, q in zip(gr.po_line_id, gr.receipt_date, gr.quantity_received):
        rec[p].append((d, q))
    df["received_as_of_invoice"] = [sum(q for d, q in rec[p] if d <= dt)
                                    for p, dt in zip(df.po_line_id, df.invoice_date)]

    docs = df.drop_duplicates("invoice_id")[["invoice_id", "supplier_id", "invoice_number", "invoice_date"]]
    docs = docs.sort_values(["invoice_date", "invoice_id"])
    key = [docs.supplier_id, docs.invoice_number]
    docs["dup_rank"] = docs.groupby(key).cumcount()
    docs["dup_size"] = docs.groupby(key).invoice_id.transform("count")
    dx = docs.set_index("invoice_id")
    df["dup_rank"] = df.invoice_id.map(dx.dup_rank)
    df["dup_size"] = df.invoice_id.map(dx.dup_size)
    df["flag_dup"] = df.dup_size > 1

    n_prior, med, ratio, prior_ids = {}, {}, {}, {}
    for _, g in df.groupby(["supplier_id", "item_id"], sort=False):
        dts, prices, lids = g.invoice_date.to_numpy(), g.price_c.to_numpy(), g.invoice_line_id.to_numpy()
        for j, ix in enumerate(g.index):
            m = dts < dts[j]
            n_prior[ix] = int(m.sum())
            prior_ids[ix] = list(lids[m])
            if m.sum() >= HIST_MIN_PRIOR:
                med[ix] = float(np.median(prices[m]))
                ratio[ix] = prices[j] / med[ix]
    df["hist_n_prior"] = pd.Series(n_prior)
    df["hist_median_c"] = pd.Series(med)
    df["hist_ratio"] = pd.Series(ratio)
    df["hist_prior_ids"] = pd.Series(prior_ids)
    df["flag_hist"] = (df.hist_n_prior >= HIST_MIN_PRIOR) & (df.hist_ratio > HIST_RATIO)
    return df.sort_values("_pos")


def stage6b_ground_truth(S, ids, T, R):
    inv, po, gr, amd = T["invoices"], T["purchase_orders"], T["goods_receipts"], T["po_amendments"]
    key_of = {v: k for k, v in ids["il"].items()}
    rc_by_pl = gr.groupby("po_line_id").receipt_id.apply(list).to_dict()
    il_by_pl = R.sort_values(["invoice_date", "invoice_line_id"]).groupby("po_line_id").invoice_line_id.apply(list).to_dict()
    amd_by_pl = amd.groupby("po_line_id").amendment_id.apply(list).to_dict()
    amdx = amd.set_index("amendment_id")
    lines_per_doc = inv.groupby("invoice_id").size()
    docs_per_po = inv.groupby("po_id").invoice_id.nunique()
    variable = {s["supplier_id"]: s["variable"] for s in S.suppliers}
    amount_c = cents(inv.line_amount)
    paid = cents(inv.amount_paid)
    p95 = amount_c.quantile(.95)
    paid_by_il = dict(zip(inv.invoice_line_id, paid))
    pox = po.set_index("po_line_id")
    fired_cols = [(DUP, "flag_dup"), (QTY, "flag_qty"), (PRICE, "flag_price"), (HIST, "flag_hist")]
    rows = []
    for i, r in enumerate(R.itertuples(index=False)):
        k = key_of[r.invoice_line_id]
        plan = S.plan.get(k)
        fired = [t for t, c in fired_cols if getattr(r, c)]
        tags = []
        if r.received_as_of_invoice < r.cum_invoiced and not r.flag_qty:
            tags.append("PARTIAL_DELIVERY_LATER_COMPLETED")
        if len(rc_by_pl.get(r.po_line_id, [])) >= 2:
            tags.append("MULTI_RECEIPT_PO_LINE")
        if r.amendment_applied:
            tags.append("APPROVED_PRICE_AMENDMENT_APPLIED")
        for a in amd_by_pl.get(r.po_line_id, []):
            if amdx.loc[a, "amendment_type"] == "QUANTITY":
                tags.append("APPROVED_QTY_AMENDMENT")
            elif amdx.loc[a, "status"] == "Rejected":
                tags.append("REJECTED_AMENDMENT_ON_FILE")
        if amount_c.iloc[i] >= p95:
            tags.append("HIGH_VALUE")
        if lines_per_doc[r.invoice_id] >= 2:
            tags.append("MULTI_LINE_INVOICE")
        if docs_per_po[r.po_id] >= 2:
            tags.append("MULTIPLE_INVOICES_PER_PO")
        if variable[r.supplier_id]:
            tags.append("VARIABLE_PRICE_SUPPLIER")
        row = dict(invoice_id=r.invoice_id, invoice_line_id=r.invoice_line_id, exposure_currency=CURRENCY,
                   duplicate_group_id="", hist_n_prior=np.nan, hist_baseline_median=np.nan,
                   hist_price_ratio=np.nan)
        if plan is None:
            row.update(is_anomaly=0, primary_anomaly_type=NORMAL,
                       all_applicable_anomaly_types="|".join(fired) if fired else NORMAL,
                       expected_exposure=0.0, exposure_paid_to_date=0.0, exposure_basis="none",
                       expected_evidence_ids="")
            desc = (f"Normal: invoiced {r.quantity_invoiced} x {usd(r.price_c)} matches the approved price "
                    f"{usd(r.approved_price_c)}; cumulative invoiced {r.cum_invoiced} <= received {r.received_total}.")
            if "PARTIAL_DELIVERY_LATER_COMPLETED" in tags:
                desc += (f" Only {r.received_as_of_invoice} received by the invoice date; later receipts "
                         f"completed the quantity, so the line reconciles.")
            if r.amendment_applied:
                desc += f" Approved amendment {r.po_amendment_id} raised the price from PO {usd(r.po_price_c)}."
            row["case_description"] = desc
        else:
            p = plan["primary"]
            tags = plan["tags"] + tags
            exp_c = plan["exposure_c"]
            pr = paid_by_il[r.invoice_line_id]
            amt = int(amount_c.iloc[i])
            row.update(is_anomaly=1, primary_anomaly_type=p,
                       all_applicable_anomaly_types="|".join([p] + [t for t in fired if t != p]))
            if p == DUP:
                partner_il = ids["il"][plan["partner"]]
                partner_inv = partner_il.rsplit("-", 1)[0]
                pp = paid_by_il[partner_il]
                first = "DUP_FIRST_ENTRY" in tags
                row["duplicate_group_id"] = f"DUPG-{plan['group']:04d}"
                ev = [r.invoice_id, partner_inv] if first else [partner_inv, r.invoice_id]
                evl = [r.invoice_line_id, partner_il] if first else [partner_il, r.invoice_line_id]
                row["expected_evidence_ids"] = ";".join(ev + evl + [r.po_line_id])
                realized = max(0, pr + pp - amt) if not first else 0
                row["exposure_paid_to_date"] = realized / 100
                row["exposure_basis"] = ("duplicate pair: exposure assigned to repeat entry" if not first
                                         else "duplicate pair: first entry carries 0 (counted on repeat)")
                row["case_description"] = (
                    f"Supplier invoice number '{r.invoice_number}' from {r.supplier_id} is entered twice "
                    f"({ev[0]} and {ev[1]}) for the same PO line, quantity and price; duplicated amount "
                    f"{usd(plan['amount_c'])}. This record is the {'first' if first else 'repeat'} entry "
                    f"(status {r.payment_status}). Potential duplicate-payment exposure {usd(plan['amount_c'])} "
                    f"is counted once, on the repeat entry. Paid across the pair: {usd(pr + pp)}; realized "
                    f"duplicate payment: {usd(max(0, pr + pp - amt))}.")
            elif p == QTY:
                rids = rc_by_pl.get(r.po_line_id, [])
                row["expected_evidence_ids"] = ";".join([r.po_line_id] + rids + il_by_pl[r.po_line_id])
                row["exposure_basis"] = "excess_quantity x approved_unit_price"
                row["exposure_paid_to_date"] = round(exp_c * pr / amt) / 100 if amt else 0.0
                how = ("supplier billed the full quantity but the undelivered balance never arrived"
                       if "UNDELIVERED_BALANCE_BILLED" in tags else "invoiced quantity exceeds what was delivered")
                row["case_description"] = (
                    f"PO line {r.po_line_id}: total received {r.received_total} across {len(rids)} receipt "
                    f"line(s) ({', '.join(rids)}); cumulative invoiced {r.cum_invoiced}. Excess {plan['excess']} "
                    f"unit(s) not covered by any receipt or approved amendment ({how}). Exposure = "
                    f"{plan['excess']} x {usd(plan['appr_c'])} = {usd(exp_c)}.")
            elif p == PRICE:
                aids = amd_by_pl.get(r.po_line_id, [])
                row["expected_evidence_ids"] = ";".join([r.po_line_id] + aids)
                row["exposure_basis"] = "(invoice_unit_price - approved_unit_price) x quantity_invoiced"
                row["exposure_paid_to_date"] = round(exp_c * pr / amt) / 100 if amt else 0.0
                note = ""
                if "ABOVE_APPROVED_AMENDMENT" in tags:
                    note = f" Approved amendment {aids[0]} was applied first; the invoice still exceeds it."
                elif "REJECTED_AMENDMENT_PRICE_BILLED" in tags:
                    note = (f" Supplier billed the price requested in amendment {aids[0]}, which was Rejected "
                            f"and therefore does not apply.")
                row["case_description"] = (
                    f"Invoiced unit price {usd(plan['new_c'])} vs approved {usd(plan['appr_c'])} (original PO "
                    f"price {usd(r.po_price_c)}).{note} Overcharge {usd(plan['new_c'] - plan['appr_c'])}/unit x "
                    f"{r.quantity_invoiced} = {usd(exp_c)}.")
            else:
                prior = [ids["il"][pk] for pk in plan["prior_keys"]]
                pp_ = R.set_index("invoice_line_id").loc[prior, "price_c"].to_numpy()
                row.update(hist_n_prior=plan["n_prior"], hist_baseline_median=round(plan["median_c"] / 100, 2),
                           hist_price_ratio=round(plan["new_c"] / plan["median_c"], 4))
                row["expected_evidence_ids"] = ";".join([r.po_line_id] + prior)
                row["exposure_paid_to_date"] = round(exp_c * pr / amt) / 100 if amt else 0.0
                base = (f"Unit price {usd(plan['new_c'])} is {plan['new_c'] / plan['median_c']:.2f}x the median "
                        f"{usd(round(plan['median_c']))} of {plan['n_prior']} prior invoice lines for the same "
                        f"supplier/item (prior range {usd(pp_.min())}-{usd(pp_.max())}, most recent "
                        f"{usd(pp_[-1])}); only earlier-dated lines used.")
                if "NO_CONTRACT_BASIS" in tags:
                    row["exposure_basis"] = "0: no defensible reference price (PO approved at this price)"
                    row["case_description"] = base + (" The PO was approved at this price, so there is no "
                                                      "contractual basis for an overcharge: refer for price review, exposure 0.")
                else:
                    row["exposure_basis"] = "(invoice_unit_price - approved_unit_price) x quantity_invoiced"
                    row["case_description"] = base + (f" The invoice also exceeds the approved PO price "
                                                      f"{usd(plan['appr_c'])}; exposure limited to that contractual "
                                                      f"excess = {usd(exp_c)}.")
            row["expected_exposure"] = exp_c / 100
        row["scenario_tags"] = ";".join(dict.fromkeys(tags))
        rows.append(row)
    cols = ["invoice_id", "invoice_line_id", "is_anomaly", "primary_anomaly_type",
            "all_applicable_anomaly_types", "expected_exposure", "exposure_currency",
            "expected_evidence_ids", "case_description", "exposure_basis", "exposure_paid_to_date",
            "duplicate_group_id", "hist_n_prior", "hist_baseline_median", "hist_price_ratio", "scenario_tags"]
    gt = pd.DataFrame(rows)[cols]
    gt["hist_n_prior"] = gt["hist_n_prior"].astype("Int64")
    return gt


# ----------------------------------------------------------------------------
# Stage 7 - validation
# ----------------------------------------------------------------------------
DATE_COLS = {"purchase_orders": ["po_date"], "goods_receipts": ["receipt_date"],
             "invoices": ["invoice_date"], "po_amendments": ["amendment_date"]}
STR_COLS = {"invoices": {"invoice_number": str}, "purchase_orders": {"approved_amendment_id": str}}
FILES = ["suppliers", "purchase_orders", "po_amendments", "goods_receipts", "invoices"]


def load_tables(d=OUT_DIR):
    T = {}
    for f in FILES:
        T[f] = pd.read_csv(os.path.join(d, f + ".csv"), parse_dates=DATE_COLS.get(f, []),
                           dtype=STR_COLS.get(f))
    return T


def stage7_validate(T, gt, R, S):
    checks = []

    def chk(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    inv, po, gr, amd, sup = (T[k] for k in ["invoices", "purchase_orders", "goods_receipts",
                                            "po_amendments", "suppliers"])
    g = gt.set_index("invoice_line_id")
    Rx = R.set_index("invoice_line_id")
    vc = gt.primary_anomaly_type.value_counts()

    # counts
    chk("Exactly 5,000 invoice lines", len(inv) == N_INVOICE_LINES, f"{len(inv)}")
    chk("Ground truth has one row per invoice line, same order",
        len(gt) == len(inv) and (gt.invoice_line_id.values == inv.invoice_line_id.values).all(), f"{len(gt)}")
    n_an = int((gt.is_anomaly == 1).sum())
    chk(f"{N_NORMAL:,} normal / {N_ANOMALIES:,} anomalous (exactly {N_ANOMALIES / N_INVOICE_LINES:.0%} anomaly rate)",
        (gt.is_anomaly == 0).sum() == N_NORMAL and n_an == N_ANOMALIES,
        f"normal={(gt.is_anomaly == 0).sum()} anomalous={n_an} rate={n_an / len(gt):.2%}")
    chk("Primary anomaly types match the configured non-uniform mix",
        all(vc.get(t, 0) == N_TYPE[t] for t in TYPES) and sum(N_TYPE.values()) == N_ANOMALIES
        and len(set(N_TYPE.values())) == len(TYPES),
        ", ".join(f"{t}={vc.get(t, 0)} ({vc.get(t, 0) / N_ANOMALIES:.0%})" for t in TYPES))
    chk("is_anomaly consistent with primary_anomaly_type",
        ((gt.is_anomaly == 1) == (gt.primary_anomaly_type != NORMAL)).all())
    # keys & document structure
    chk("invoice_line_id unique", inv.invoice_line_id.is_unique)
    for t, c in [("suppliers", "supplier_id"), ("purchase_orders", "po_line_id"),
                 ("goods_receipts", "receipt_id"), ("po_amendments", "amendment_id")]:
        chk(f"{c} unique in {t}", T[t][c].is_unique)
    chk("Supplier IDs follow SUP-0001 pattern, 80 suppliers",
        len(sup) == N_SUPPLIERS and sup.supplier_id.str.fullmatch(r"SUP-\d{4}").all())
    chk("invoice_line_id is prefixed by its invoice_id",
        (inv.invoice_line_id.str.rsplit("-", n=1).str[0] == inv.invoice_id).all())
    doc_consistent = inv.groupby("invoice_id")[["supplier_id", "invoice_number", "invoice_date",
                                                 "payment_status", "po_id"]].nunique().max().max() == 1
    chk("Each invoice document has one supplier, number, date, status and PO", doc_consistent)
    # referential integrity
    chk("Invoice supplier_id exists in suppliers", inv.supplier_id.isin(sup.supplier_id).all())
    chk("Invoice po_line_id exists in purchase_orders", inv.po_line_id.isin(po.po_line_id).all())
    chk("Invoice item_id exists in the catalog", inv.item_id.isin([i["item_id"] for i in S.items]).all())
    chk("PO supplier_id exists in suppliers", po.supplier_id.isin(sup.supplier_id).all())
    chk("Receipt po_line_id exists in purchase_orders", gr.po_line_id.isin(po.po_line_id).all())
    chk("Amendment po_line_id exists in purchase_orders", amd.po_line_id.isin(po.po_line_id).all())
    pa = po[po.approved_amendment_id.notna()].merge(amd, left_on="approved_amendment_id",
                                                    right_on="amendment_id", suffixes=("", "_a"))
    chk("PO approved_amendment_id resolves to an Approved amendment of the same PO line",
        len(pa) == po.approved_amendment_id.notna().sum() and (pa.status == "Approved").all()
        and (pa.po_line_id == pa.po_line_id_a).all(), f"{len(pa)} linked amendments")
    chk("Approved amendments always referenced by their PO line",
        amd[amd.status == "Approved"].amendment_id.isin(po.approved_amendment_id).all())
    pox = po.set_index("po_line_id")
    chk("Invoice supplier matches PO supplier", (inv.po_line_id.map(pox.supplier_id) == inv.supplier_id).all())
    chk("Invoice item matches PO line item", (inv.po_line_id.map(pox.item_id) == inv.item_id).all())
    chk("Invoice po_id matches PO line's po_id", (inv.po_line_id.map(pox.po_id) == inv.po_id).all())
    chk("Receipt item and po_id match PO line", (gr.po_line_id.map(pox.item_id) == gr.item_id).all()
        and (gr.po_line_id.map(pox.po_id) == gr.po_id).all())
    # values
    chk("All quantities positive", (inv.quantity_invoiced > 0).all() and (gr.quantity_received > 0).all()
        and (po.quantity_ordered > 0).all())
    chk("All prices positive", (inv.unit_price > 0).all() and (po.unit_price > 0).all())
    chk("Single currency USD", set(inv.currency) == {CURRENCY} and set(po.currency) == {CURRENCY}
        and set(gt.exposure_currency) == {CURRENCY})
    chk("Monetary values have at most 2 decimals",
        all(np.allclose(x * 100, np.round(x * 100), atol=1e-6) for x in
            [inv.unit_price, inv.line_amount, inv.amount_paid, po.unit_price, gt.expected_exposure]))
    chk("line_amount = quantity_invoiced x unit_price", (cents(inv.line_amount) ==
                                                         inv.quantity_invoiced * cents(inv.unit_price)).all())
    # dates
    po_date = gr.po_line_id.map(pox.po_date)
    chk("Receipt dates not before PO date", (gr.receipt_date >= po_date).all())
    chk("Invoice dates not before PO date", (inv.invoice_date >= inv.po_line_id.map(pox.po_date)).all())
    first_rc_po = gr.groupby("po_id").receipt_date.min()
    chk("Invoice dates not before the PO's first delivery", (inv.invoice_date >= inv.po_id.map(first_rc_po)).all())
    first_inv_pl = inv.groupby("po_line_id").invoice_date.min()
    fi = amd.po_line_id.map(first_inv_pl)
    chk("Amendment dates between PO date and first invoice of the line",
        (amd.amendment_date >= amd.po_line_id.map(pox.po_date)).all() and (fi.isna() | (amd.amendment_date <= fi)).all())
    lo, hi = pd.Timestamp(START), pd.Timestamp(AS_OF)
    chk("All dates inside the 12-month window",
        all(((s >= lo) & (s <= hi)).all() for s in [po.po_date, gr.receipt_date, inv.invoice_date]),
        f"{inv.invoice_date.min().date()} .. {inv.invoice_date.max().date()}")
    # payments
    amt, paid = cents(inv.line_amount), cents(inv.amount_paid)
    chk("Paid lines: amount_paid = line_amount", (paid[inv.payment_status == "Paid"] == amt[inv.payment_status == "Paid"]).all())
    chk("Unpaid lines: amount_paid = 0", (paid[inv.payment_status == "Unpaid"] == 0).all())
    pp = inv.payment_status == "Partially Paid"
    chk("Partially Paid lines: 0 < amount_paid < line_amount", ((paid[pp] > 0) & (paid[pp] < amt[pp])).all())
    an = inv[gt.is_anomaly.values == 1]
    chk("Flagged invoices are not all paid (all three statuses present among anomalies)",
        set(an.payment_status) == {"Paid", "Partially Paid", "Unpaid"}, an.payment_status.value_counts().to_dict())

    # rule-based label checks
    for t, c in [(DUP, "flag_dup"), (QTY, "flag_qty"), (PRICE, "flag_price"), (HIST, "flag_hist")]:
        ids_t = gt.invoice_line_id[gt.primary_anomaly_type == t]
        chk(f"{t}: every planted record fires its reference rule", Rx.loc[ids_t, c].all(),
            f"{int(Rx.loc[ids_t, c].sum())}/{len(ids_t)}")
    normal_ids = gt.invoice_line_id[gt.is_anomaly == 0]
    fired_normal = Rx.loc[normal_ids, ["flag_dup", "flag_qty", "flag_price", "flag_hist"]].any(axis=1)
    chk("No normal record fires any of the four rules", not fired_normal.any(), f"{int(fired_normal.sum())} violations")
    chk("primary type contained in all_applicable_anomaly_types",
        all(p in a.split("|") for p, a in zip(gt.primary_anomaly_type, gt.all_applicable_anomaly_types)))
    chk("Normal records have all_applicable = NORMAL and exposure 0",
        (gt[gt.is_anomaly == 0].all_applicable_anomaly_types == NORMAL).all()
        and (gt[gt.is_anomaly == 0].expected_exposure == 0).all())
    fired_any = Rx[["flag_dup", "flag_qty", "flag_price", "flag_hist"]].any(axis=1)
    chk("Every line that fires a rule is a labelled anomaly", (gt.set_index("invoice_line_id").loc[fired_any[fired_any].index, "is_anomaly"] == 1).all())

    # duplicates
    d = gt[gt.primary_anomaly_type == DUP].merge(inv, on=["invoice_id", "invoice_line_id"])
    grp = d.groupby(["supplier_id", "invoice_number"])
    chk(f"{N_DUP_PAIRS} duplicate pairs detectable by key (supplier_id, invoice_number)",
        grp.ngroups == N_DUP_PAIRS and (grp.invoice_id.nunique() == 2).all(), f"{grp.ngroups} groups")
    keys_all = inv.drop_duplicates("invoice_id").groupby(["supplier_id", "invoice_number"]).size()
    chk("No unplanned invoice-number repeats in the data", (keys_all > 1).sum() == N_DUP_PAIRS)
    chk("Duplicate pairs share PO line, item, quantity and price",
        (grp[["po_line_id", "item_id", "quantity_invoiced", "unit_price"]].nunique().max().max() == 1))
    chk("Duplicate exposure counted once per pair",
        (d.groupby("duplicate_group_id").expected_exposure.apply(lambda s: (s > 0).sum()) == 1).all()
        and np.allclose(d.groupby("duplicate_group_id").expected_exposure.sum().values,
                        d.groupby("duplicate_group_id").line_amount.first().values))
    combos = d.sort_values("invoice_date").groupby("duplicate_group_id").payment_status.apply(tuple).value_counts()
    chk("Duplicate pairs cover several payment-status combinations", len(combos) >= 4,
        "; ".join(f"{a}/{b}: {n}" for (a, b), n in combos.items()))

    # quantity
    q_ids = gt.invoice_line_id[gt.primary_anomaly_type == QTY]
    chk("Quantity discrepancies persist after aggregating ALL receipts (incl. later ones)",
        (Rx.loc[q_ids, "cum_invoiced"] > Rx.loc[q_ids, "received_total"]).all())
    rc_count = gr.groupby("po_line_id").size()
    chk("Some quantity discrepancies involve multiple receipt lines",
        (Rx.loc[q_ids, "po_line_id"].map(rc_count) >= 2).sum() > 0,
        f"{int((Rx.loc[q_ids, 'po_line_id'].map(rc_count) >= 2).sum())} cases")
    chk("Quantity cases list their receipt IDs as evidence",
        all(all(r in g.loc[i, "expected_evidence_ids"].split(";") for r in gr.receipt_id[gr.po_line_id == Rx.loc[i, "po_line_id"]])
            for i in q_ids))
    # price
    p_ids = gt.invoice_line_id[gt.primary_anomaly_type == PRICE]
    chk("PO price mismatches remain after applying approved amendments",
        (Rx.loc[p_ids, "price_c"] > Rx.loc[p_ids, "approved_price_c"]).all(),
        f"{int(Rx.loc[p_ids, 'amendment_applied'].sum())} cases had an approved amendment applied first")
    exc = Rx.loc[p_ids, "price_c"] / Rx.loc[p_ids, "approved_price_c"] - 1
    chk("Price mismatches include modest (<8%) and substantial (>=20%) overcharges",
        (exc < .08).sum() > 0 and (exc >= .20).sum() > 0,
        f"modest={int((exc < .08).sum())}, moderate={int(((exc >= .08) & (exc < .2)).sum())}, substantial={int((exc >= .2).sum())}")
    # historical
    h_ids = gt.invoice_line_id[gt.primary_anomaly_type == HIST]
    chk("Historical spikes have >= 5 strictly-earlier lines for the same supplier/item",
        (Rx.loc[h_ids, "hist_n_prior"] >= HIST_MIN_PRIOR).all(),
        f"min prior lines = {int(Rx.loc[h_ids, 'hist_n_prior'].min())}, median = {int(Rx.loc[h_ids, 'hist_n_prior'].median())}")
    chk("Historical baselines use only past data (ground-truth median = rule median on prior lines)",
        np.allclose(g.loc[h_ids, "hist_baseline_median"].astype(float).values,
                    (Rx.loc[h_ids, "hist_median_c"] / 100).values, atol=0.0051, rtol=0))
    zero = (g.loc[h_ids, "expected_exposure"] == 0).sum()
    chk("Some historical spikes carry zero exposure (no contractual basis)", 0 < zero < N_TYPE[HIST],
        f"{int(zero)} zero-exposure, {N_TYPE[HIST] - int(zero)} with PO-based exposure")
    # normal scenarios
    nrm = Rx.loc[normal_ids]
    partial = nrm[nrm.received_as_of_invoice < nrm.cum_invoiced]
    chk("Normal partial deliveries later completed exist and are labelled normal",
        len(partial) > 0 and (partial.cum_invoiced <= partial.received_total).all(), f"{len(partial)} lines")
    amended = nrm[nrm.amendment_applied & (nrm.price_c > nrm.po_price_c)]
    chk("Normal lines priced above original PO via approved amendment exist and are not flagged",
        len(amended) > 0 and not amended.flag_price.any(), f"{len(amended)} lines")
    rej = gt.set_index("invoice_line_id").loc[normal_ids].scenario_tags.str.contains("REJECTED_AMENDMENT_ON_FILE")
    chk("Normal lines with a rejected amendment on file are billed at the original price", rej.sum() > 0 and
        (Rx.loc[rej[rej].index, "price_c"] == Rx.loc[rej[rej].index, "po_price_c"]).all(), f"{int(rej.sum())} lines")
    for st in ["Paid", "Partially Paid", "Unpaid"]:
        chk(f"Normal lines include '{st}' invoices", (inv[gt.is_anomaly.values == 0].payment_status == st).sum() > 0)
    hv = gt[gt.scenario_tags.str.contains("HIGH_VALUE")]
    chk("High-value lines are mixed normal/anomalous", hv.is_anomaly.nunique() == 2,
        f"{(hv.is_anomaly == 0).sum()} normal, {(hv.is_anomaly == 1).sum()} anomalous")
    chk("Normal lines include multi-line invoices and multi-invoice POs",
        gt[gt.is_anomaly == 0].scenario_tags.str.contains("MULTI_LINE_INVOICE").any()
        and gt[gt.is_anomaly == 0].scenario_tags.str.contains("MULTIPLE_INVOICES_PER_PO").any())
    # exposure recomputation from tables
    rec_c = []
    for i, row in gt.iterrows():
        r = Rx.loc[row.invoice_line_id]
        t = row.primary_anomaly_type
        if t == NORMAL:
            v = 0
        elif t == DUP:
            v = r.price_c * r.quantity_invoiced if r.dup_rank > 0 else 0
        elif t == QTY:
            v = r.excess_qty * r.approved_price_c
        else:
            v = r.price_excess_c * r.quantity_invoiced
        rec_c.append(int(v))
    gt_c = cents(gt.expected_exposure)
    mism = int((gt_c.values != np.array(rec_c)).sum())
    chk("expected_exposure matches the documented formulas recomputed from the tables", mism == 0,
        f"{mism} mismatches; total exposure {usd(int(gt_c.sum()))}")
    # leakage
    gt_cols = set(gt.columns) - {"invoice_id", "invoice_line_id"}
    bad = [f"{t}.{c}" for t in FILES for c in T[t].columns
           if c in gt_cols or any(w in c.lower() for w in ["anomal", "label", "fraud", "flag", "is_", "ground", "truth", "planted"])]
    chk("No ground-truth or label-like columns in procurement tables", not bad, ", ".join(bad) or "clean")
    lab = gt.is_anomaly.values
    month = inv.invoice_date.dt.to_period("M").astype(str)
    fmt = inv.invoice_number.str.replace(r"\d", "9", regex=True)
    pure = []
    for name, s in [("supplier_id", inv.supplier_id), ("month", month), ("payment_status", inv.payment_status),
                    ("invoice_number_format", fmt), ("item_id", inv.item_id)]:
        rate = pd.Series(lab).groupby(s.values).agg(["mean", "size"])
        if ((rate["size"] >= 10) & (rate["mean"] == 1)).any():
            pure.append(name)
    chk("No supplier/month/status/number-format/item value is exclusively anomalous (n>=10)", not pure,
        ", ".join(pure) or "none")
    chk("Anomalies spread across suppliers and months",
        inv.supplier_id[lab == 1].nunique() >= 60 and month[lab == 1].nunique() >= 10,
        f"{inv.supplier_id[lab == 1].nunique()} suppliers, {month[lab == 1].nunique()} months")
    dec = pd.Series(lab).groupby(np.arange(len(lab)) * 10 // len(lab)).mean()
    chk("Anomalies interleaved through file order (every decile 3-20% anomalous)",
        dec.between(.03, .20).all(), " ".join(f"{x:.0%}" for x in dec))
    return checks


# ----------------------------------------------------------------------------
# Stage 8/9 - save + report
# ----------------------------------------------------------------------------
def save_csv(df, name):
    df.to_csv(os.path.join(OUT_DIR, name + ".csv"), index=False, float_format="%.2f",
              date_format="%Y-%m-%d")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def stage9_report(T, gt, checks, S):
    inv = T["invoices"]
    m = gt.merge(inv[["invoice_line_id", "payment_status", "line_amount"]], on="invoice_line_id")
    L = ["# AI Leak Hunter synthetic dataset: validation report", "",
         f"Seed {SEED}. Activity window {START} to {AS_OF} (as-of date for payments).", "",
         "## Validation checks", "", "| # | Check | Result | Detail |", "|---|---|---|---|"]
    for i, (n, ok, det) in enumerate(checks, 1):
        L.append(f"| {i} | {n} | {'PASS' if ok else '**FAIL**'} | {str(det).replace('|', '/')} |")
    passed = sum(ok for _, ok, _ in checks)
    L += ["", f"**{passed}/{len(checks)} checks passed.**", "", "## Table sizes", "",
          "| File | Rows |", "|---|---|"]
    for f in FILES:
        L.append(f"| {f}.csv | {len(T[f]):,} |")
    L.append(f"| ground_truth.csv | {len(gt):,} |")
    po = T["purchase_orders"]
    L += ["", f"{po.po_id.nunique():,} purchase orders, {inv.invoice_id.nunique():,} invoice documents, "
          f"{inv.item_id.nunique()} distinct items invoiced across {len(CATALOG)} categories, "
          f"{inv.supplier_id.nunique()} suppliers. Supplier tiers: "
          + ", ".join(f"{k} {v}" for k, v in T['suppliers'].supplier_tier.value_counts().items()) + ".",
          f"PO amendments: " + ", ".join(f"{a} {s}: {n}" for (a, s), n in
                                         T['po_amendments'].groupby(['amendment_type', 'status']).size().items()) + ".",
          f"Ordinary price normalisations applied during generation: {S.stats['normal_price_repairs']}.", "",
          "## Anomaly distribution", "", "| Primary type | Records | Expected exposure | Mean | Max | Exposure paid to date |",
          "|---|---|---|---|---|---|"]
    for t in [NORMAL] + TYPES:
        s = m[m.primary_anomaly_type == t]
        L.append(f"| {t} | {len(s)} | ${s.expected_exposure.sum():,.2f} | ${s.expected_exposure.mean():,.2f} | "
                 f"${s.expected_exposure.max():,.2f} | ${s.exposure_paid_to_date.sum():,.2f} |")
    L.append(f"| **Total** | {len(m)} | ${m.expected_exposure.sum():,.2f} | | | ${m.exposure_paid_to_date.sum():,.2f} |")
    L += ["", "### All applicable labels (primary first)", "", "| all_applicable_anomaly_types | Records |", "|---|---|"]
    for k, v in m[m.is_anomaly == 1].all_applicable_anomaly_types.value_counts().items():
        L.append(f"| {k} | {v} |")
    L += ["", "### Planting variants", "", "| Primary type | Variant | Records |", "|---|---|---|"]
    variants = {DUP: ["DUP_FIRST_ENTRY", "DUP_REPEAT_ENTRY"],
                QTY: ["INVOICE_QTY_INFLATED", "UNDELIVERED_BALANCE_BILLED", "EXCESS_SMALL", "EXCESS_MEDIUM", "EXCESS_LARGE"],
                PRICE: ["NO_AMENDMENT", "ABOVE_APPROVED_AMENDMENT", "REJECTED_AMENDMENT_PRICE_BILLED",
                        "OVERCHARGE_MODEST", "OVERCHARGE_MODERATE", "OVERCHARGE_SUBSTANTIAL"],
                HIST: ["NO_CONTRACT_BASIS", "ALSO_ABOVE_PO_PRICE", "SPIKE_MODERATE", "SPIKE_HIGH", "SPIKE_EXTREME"]}
    for t, vs in variants.items():
        s = m[m.primary_anomaly_type == t]
        for v in vs:
            L.append(f"| {t} | {v} | {int(s.scenario_tags.str.split(';').apply(lambda x: v in x).sum())} |")
    L += ["", "### Payment status by primary type", "", "| Primary type | Paid | Partially Paid | Unpaid |", "|---|---|---|---|"]
    for t in [NORMAL] + TYPES:
        c = m[m.primary_anomaly_type == t].payment_status.value_counts()
        L.append(f"| {t} | {c.get('Paid', 0)} | {c.get('Partially Paid', 0)} | {c.get('Unpaid', 0)} |")
    L += ["", "### Normal scenario coverage", "", "| Scenario tag | Normal records |", "|---|---|"]
    tags = Counter(t for s in m[m.is_anomaly == 0].scenario_tags for t in s.split(";") if t)
    for k, v in sorted(tags.items()):
        L.append(f"| {k} | {v} |")
    L += ["", "### Anomaly share by invoice month (leakage diagnostic)", "", "| Month | Lines | Anomalous share |", "|---|---|---|"]
    mo = inv.invoice_date.dt.to_period("M").astype(str)
    for k, s in pd.Series(gt.is_anomaly.values).groupby(mo.values):
        L.append(f"| {k} | {len(s)} | {s.mean():.0%} |")
    L += ["", "Historical-spike anomalies need at least five earlier purchases of the same item from the same "
          "supplier, so they cannot occur in the first months; the other types are spread across the year.",
          "", "## File fingerprints (SHA-256, first 16 hex)", ""]
    for f in FILES + ["ground_truth"]:
        L.append(f"- {f}.csv: `{sha(os.path.join(OUT_DIR, f + '.csv'))}`")
    open(os.path.join(OUT_DIR, "validation_report.md"), "w").write("\n".join(L) + "\n")
    return passed


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    S = State()
    print("[1] suppliers & catalog");   stage1_suppliers_and_catalog(S)
    print("[2] purchase orders");       stage2_purchase_orders(S)
    print("[3] goods receipts");        stage3_goods_receipts(S)
    print("[4] ordinary invoices");     stage4_invoices(S); stage4b_trim(S); stage4c_normalise_prices(S)
    print("[5] anomaly injection");     stage5_inject_anomalies(S); stage5b_payments(S); stage5c_invoice_numbers(S)
    print("[6] tables & ground truth")
    T, ids = stage6_build_tables(S)
    print("[8] saving procurement CSVs")
    for name, df in T.items():
        save_csv(df, name)
    T = load_tables()                       # rules run on what the pipeline will actually load
    R = reference_rules(T["purchase_orders"], T["goods_receipts"], T["invoices"], T["po_amendments"])
    gt = stage6b_ground_truth(S, ids, T, R)
    save_csv(gt, "ground_truth")
    gt = pd.read_csv(os.path.join(OUT_DIR, "ground_truth.csv"), keep_default_na=False,
                     na_values={"hist_n_prior": [""], "hist_baseline_median": [""], "hist_price_ratio": [""]})
    print("[7] validation")
    checks = stage7_validate(T, gt, R, S)
    print("[9] report")
    passed = stage9_report(T, gt, checks, S)
    width = max(len(n) for n, _, _ in checks)
    for n, ok, det in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {n:<{width}}  {det}")
    print(f"\n{passed}/{len(checks)} checks passed.")
    print(gt.primary_anomaly_type.value_counts().to_string())
    print(f"Total expected exposure: ${gt.expected_exposure.sum():,.2f}")
    print(f"Files written to {OUT_DIR}")
    if passed != len(checks):
        sys.exit("Validation failed - fix the generator and re-run.")


if __name__ == "__main__":
    main()
