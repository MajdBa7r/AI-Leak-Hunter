# AI Leak Hunter: Synthetic Procurement Dataset

A reproducible, fully fictional procurement dataset for building and evaluating the AI Leak Hunter anomaly-detection pipeline: 5,000 invoice lines (4,500 normal, 500 planted anomalies, a 10% anomaly rate, split unevenly across four categories) with linked purchase orders, amendments, goods receipts and suppliers, plus an evaluation-only ground-truth file.

```
generate_synthetic_data.py      # regenerates everything below (seed 42, ~5 s)
data/
  suppliers.csv                 # 80 suppliers
  purchase_orders.csv           # 4,425 PO lines (2,218 POs)
  po_amendments.csv             # supporting table: approved/rejected price & quantity amendments
  goods_receipts.csv            # 5,541 receipt lines
  invoices.csv                  # exactly 5,000 invoice lines (2,670 invoice documents)
  ground_truth.csv              # EVALUATION ONLY, one row per invoice line
  validation_report.md          # every validation check + full distribution summary
```

Regenerate with `python generate_synthetic_data.py` (Python 3.9+, pandas, numpy; no network or paid services). Runs are byte-identical; `validation_report.md` lists SHA-256 fingerprints so you can confirm. The script exits with an error if any validation check fails.

---

## 1. Anomaly distribution

| Primary type | Records | Expected exposure | How it was planted |
|---|---|---|---|
| NORMAL | 4,500 (90%) | $0 | Ordinary transactions incl. partial deliveries, approved amendments, high-value orders, variable-price suppliers |
| DUPLICATE_INVOICE | 80 (40 pairs; 1.6%) | $206,875.13 | A single-line invoice re-entered under a new internal `invoice_id`, same supplier invoice number, PO line, quantity and price, 1–30 days later |
| QUANTITY_DISCREPANCY | 140 (2.8%) | $206,684.46 | 84 invoice quantities inflated; 56 cases billed in full while the final delivery never arrived (receipts short). Excess size small / medium / large |
| PO_PRICE_MISMATCH | 170 (3.4%) | $167,642.37 | 136 with no amendment, 17 billed above an approved amendment's revised price, 17 billed at the price of a **rejected** amendment request. Modest (<8%), moderate, substantial (≥20%) |
| HISTORICAL_PRICE_SPIKE | 110 (2.2%) | $239,122.41 | 1.4×–3.0× the median of ≥5 earlier purchases of the same item from the same supplier. 77 have exposure 0 (PO approved at the high price, so no contractual basis); 33 also exceed the PO price, exposure = that contractual excess |
| **Total** | **5,000** | **$820,324.37** | |

The category mix (price 34%, quantity 28%, historical 22%, duplicates 16% of anomalies) is a design choice for a varied, non-uniform evaluation set, not an estimate of real-world prevalence. Three-way-match exceptions are the most frequent planted cases, historical spikes are fewer because each needs a five-purchase baseline, and duplicates come in pairs. Within each category the sub-variant shares (undelivered vs inflated quantity, amendment traps, zero-exposure spikes) are unchanged from the previous version. Counts are set in `N_TYPE` at the top of the script.

Secondary labels (`all_applicable_anomaly_types`, primary first): every duplicate *repeat* entry also exceeds the received quantity (`DUPLICATE_INVOICE|QUANTITY_DISCREPANCY`, 40); the 33 spikes above PO price also carry `PO_PRICE_MISMATCH`; 15 large PO overcharges also trip the historical rule. Payment statuses are drawn by invoice age for all lines, so anomalous lines are paid, partially paid and unpaid at the same rates as normal ones. Full breakdowns (variants, payment status by type, monthly shares, normal-scenario coverage) are in `validation_report.md`.

## 2. Reference rule definitions (what "anomalous" means)

These are the ground-truth definitions. The validator recomputes them from the procurement tables alone and requires every planted record to fire its rule and **no** normal record to fire any rule.

| Rule | Definition |
|---|---|
| Duplicate | More than one internal invoice document shares the matching key **(`supplier_id`, `invoice_number`)**. The earlier entry (by `invoice_date`, then `invoice_id`) is the first entry; later ones are repeats. |
| Quantity | Per `po_line_id`, order invoice lines by (`invoice_date`, `invoice_line_id`) and take the cumulative invoiced quantity. A line is flagged when that cumulative quantity exceeds the **sum of all receipt lines** for the PO line (receipts dated after the invoice included). Excess = min(line quantity, cumulative − received). |
| PO price | Approved price = PO `unit_price`, replaced by the `new_unit_price` of the PO line's `approved_amendment_id` (status Approved, type PRICE) when `amendment_date ≤ invoice_date`. Flag when invoice `unit_price` > approved price. Rejected amendments never apply. |
| Historical | For the same `supplier_id` + `item_id`, take invoice lines with `invoice_date` **strictly earlier** than this line. With ≥5 such lines, flag when `unit_price` > 1.30 × their median. No future data is used. |

## 3. Exposure conventions

| Type | `expected_exposure` | Notes |
|---|---|---|
| Normal | 0 | |
| Duplicate | Repeat entry: the line amount (potential duplicate payment). First entry: 0 | Counted once per pair. `exposure_paid_to_date` on the repeat = max(0, paid on both − line amount), i.e. a realized duplicate payment only when payments prove it. |
| Quantity | excess_quantity × approved unit price | |
| PO price | (invoice price − approved price) × quantity invoiced | |
| Historical | 0 when the PO itself carries the high price; otherwise the PO-price excess | Price-history evidence is in `case_description`, `hist_*` columns and `expected_evidence_ids`. |

All money is USD, computed in integer cents and written with two decimals. `exposure_paid_to_date` (non-duplicates) = exposure × amount_paid / line_amount, rounded to cents.

## 4. Data dictionary

### suppliers.csv
| Column | Description |
|---|---|
| supplier_id | Stable ID `SUP-0001`…`SUP-0080` |
| supplier_name | Fictional company name |
| category | One of 8 categories (IT Hardware, Office Supplies, MRO & Tools, Packaging, Lab & Chemicals, Electrical Components, Safety & PPE, Facilities Consumables) |
| country | Supplier country |
| supplier_tier | Small / Medium / Large; drives order frequency and order size (extra column) |

### purchase_orders.csv (one row per PO line)
| Column | Description |
|---|---|
| po_id | `PO-000001`, numbered chronologically |
| po_line_id | `{po_id}-NN` |
| supplier_id | FK → suppliers |
| po_date | PO date (2025-10-01 to 2026-07-01) |
| item_id | `ITM-0001`…`ITM-0080` (80 items, 10 per category) |
| item_description | Item name |
| quantity_ordered | **Original** ordered quantity (a quantity amendment may revise it) |
| unit_price | **Original** PO unit price, USD |
| currency | `USD` |
| approved_amendment_id | FK → po_amendments when an **approved** price or quantity amendment applies; empty otherwise (intentional) |

### po_amendments.csv (supporting table, added so amendments can be applied)
| Column | Description |
|---|---|
| amendment_id | `AMD-00001` |
| po_id, po_line_id | Amended PO line |
| amendment_date | Effective date (between PO date and first invoice of the line) |
| amendment_type | `PRICE` or `QUANTITY` |
| status | `Approved` or `Rejected` (rejected requests are never referenced from the PO) |
| previous_unit_price, new_unit_price | PRICE amendments only; empty for QUANTITY |
| previous_quantity, new_quantity | QUANTITY amendments only; empty for PRICE |
| decided_by | Approver role |

### goods_receipts.csv (one row per receipt line)
| Column | Description |
|---|---|
| receipt_id | `GR-000001`, chronological, unique per receipt line |
| po_id, po_line_id | FK → purchase_orders |
| receipt_date | Delivery date (≥ PO date) |
| item_id | Same as the PO line |
| quantity_received | Positive quantity; a PO line may have 1–3 receipt lines |

### invoices.csv (exactly 5,000 rows)
| Column | Description |
|---|---|
| invoice_id | Internal invoice document ID `INV-000001`, chronological; distinct for duplicate re-entries |
| invoice_number | Supplier's own number; format varies by supplier (`INV-2026-012345`, `AB0012345`, `SI/26/012345`, `10012345`, `202604-012345`). Unique per supplier except the 40 planted pairs. Read as string. |
| invoice_line_id | `{invoice_id}-NN` |
| supplier_id, po_id, po_line_id, item_id | FKs; consistent with the PO line |
| invoice_date | Invoice date (≥ PO date and ≥ the PO's first delivery) |
| quantity_invoiced | Positive |
| unit_price | Invoiced unit price, USD |
| currency | `USD` |
| line_amount | quantity_invoiced × unit_price (extra column) |
| payment_status | `Paid`, `Partially Paid`, `Unpaid` as of 2026-09-30; same for every line of a document |
| amount_paid | Paid = line_amount; Unpaid = 0; Partially Paid = 25–75% of line_amount |

### ground_truth.csv (evaluation only, one row per invoice line, same order as invoices.csv)
| Column | Description |
|---|---|
| invoice_id, invoice_line_id | Join keys |
| is_anomaly | 1 for the 500 anomalies, 0 for the 4,500 normal lines |
| primary_anomaly_type | `NORMAL`, `DUPLICATE_INVOICE`, `QUANTITY_DISCREPANCY`, `PO_PRICE_MISMATCH`, `HISTORICAL_PRICE_SPIKE` |
| all_applicable_anomaly_types | `|`-separated; primary first, then every other reference rule that fires; `NORMAL` for normal rows |
| expected_exposure | USD, per section 3 |
| exposure_currency | `USD` |
| expected_evidence_ids | `;`-separated IDs a good explanation should cite. Duplicate: both invoice IDs, both line IDs, PO line. Quantity: PO line, all its receipt IDs, all its invoice lines. PO price: PO line plus any amendment IDs. Historical: PO line plus every prior invoice line in the baseline. Empty for normal rows |
| case_description | Plain-language explanation with the numbers behind the label |
| exposure_basis | Which formula produced the exposure |
| exposure_paid_to_date | Portion of the exposure already paid (realized), USD |
| duplicate_group_id | `DUPG-0001`… for duplicate pairs; empty otherwise |
| hist_n_prior, hist_baseline_median, hist_price_ratio | Baseline evidence for historical spikes; empty otherwise |
| scenario_tags | `;`-separated slices for evaluation, e.g. `PARTIAL_DELIVERY_LATER_COMPLETED`, `APPROVED_PRICE_AMENDMENT_APPLIED`, `REJECTED_AMENDMENT_ON_FILE`, `HIGH_VALUE`, `VARIABLE_PRICE_SUPPLIER`, `MULTI_LINE_INVOICE`, plus planting variants (`UNDELIVERED_BALANCE_BILLED`, `OVERCHARGE_SUBSTANTIAL`, `NO_CONTRACT_BASIS`, …) |

Empty cells are always intentional: `approved_amendment_id` without an approved amendment, the price/quantity fields that don't apply to an amendment type, and ground-truth fields that don't apply to a row's type.

## 5. Loading into the detection pipeline

```python
import pandas as pd

DATA = "data"
dates = {"purchase_orders": ["po_date"], "goods_receipts": ["receipt_date"],
         "invoices": ["invoice_date"], "po_amendments": ["amendment_date"]}
dtypes = {"invoices": {"invoice_number": str}, "purchase_orders": {"approved_amendment_id": str}}

tables = {name: pd.read_csv(f"{DATA}/{name}.csv", parse_dates=dates.get(name, []),
                            dtype=dtypes.get(name))
          for name in ["suppliers", "purchase_orders", "po_amendments", "goods_receipts", "invoices"]}

# --- detection: use ONLY `tables` ---
# predictions = leak_hunter.run(**tables)
#   -> DataFrame[invoice_line_id, predicted_type, predicted_exposure, evidence_ids]

# --- evaluation: load labels only here ---
gt = pd.read_csv(f"{DATA}/ground_truth.csv", keep_default_na=False,
                 na_values={c: [""] for c in ["hist_n_prior", "hist_baseline_median", "hist_price_ratio"]})
# ev = gt.merge(predictions, on="invoice_line_id", how="left")
```

Recommended evaluation:

- **Detection:** precision / recall / F1 on `is_anomaly`; per-type recall on `primary_anomaly_type`; for multi-label detectors, compare against `all_applicable_anomaly_types`.
- **False-positive traps:** report false-positive rates on normal rows tagged `PARTIAL_DELIVERY_LATER_COMPLETED`, `APPROVED_PRICE_AMENDMENT_APPLIED`, `REJECTED_AMENDMENT_ON_FILE`, `HIGH_VALUE` and `VARIABLE_PRICE_SUPPLIER`. A naive point-in-time receipt check or a "price above original PO" check will fail these.
- **Exposure:** compare summed predicted vs expected exposure, and absolute error per line; for duplicates, sum per `duplicate_group_id` so a pair isn't counted twice. Report `exposure_paid_to_date` separately as realized leakage.
- **Explanations:** overlap of cited IDs with `expected_evidence_ids`.
- **Temporal honesty:** if you train a model, split by `invoice_date` (e.g. train on Oct–Apr, test on May–Sep) and build any price baselines from earlier-dated lines only, as the reference rule does.

`generate_synthetic_data.py` contains `reference_rules()`, a rule-based baseline you can import and score as a sanity floor.

## 6. Assumptions

- Reconciliation is as of the extraction date 2026-09-30; all receipts in the file count, including those dated after the invoice. Returns and credit notes are not modeled, so no receipt is negative.
- Billing patterns: per shipment, billed in full at the first delivery (later shipments complete the order), or billed after the final delivery. Every invoice is dated on or after the PO's first delivery.
- Some normal PO lines are short-shipped and billed for exactly what arrived.
- Amendments are dated before any invoice on their PO line; a price amendment applies to invoices dated on/after its `amendment_date`.
- No price tolerance is applied: any excess of one cent or more over the approved price is a mismatch.
- Historical baselines are invoice prices (not PO prices) for the same supplier and item, include every earlier line (anomalous ones too, as a real system would), and use the median for robustness. Normal prices were generated so they never exceed 1.25× their baseline, leaving a margin below the 1.30 threshold.
- Duplicate re-entries copy single-line invoices so each pair is exactly two invoice lines.
- Payment status is drawn per document from invoice age (older invoices are more likely paid), independent of labels.

## 7. Limitations of synthetic data for real-world evaluation

- **Anomalies follow the rules that define them.** Labels are clean and every anomaly is detectable by construction; real leakage is ambiguous, partially documented and often only confirmed by audit. Scores here are an upper bound.
- **Simplified duplicates.** Real duplicates often differ slightly (reformatted numbers like `INV-123` vs `INV0123`, changed dates or amounts, a different supplier record for the same vendor). Here the key matches exactly, so fuzzy-matching ability is not tested.
- **Narrow anomaly menu.** Missing receipts, unmatched invoices, split purchases, ghost suppliers, contract-rate breaches, tax/freight errors, currency effects and collusion are absent. A detector tuned here may miss them.
- **Base rate is a design choice.** 10% of lines are anomalous. That is far closer to an imbalanced real-world setting than an even split, but it was not calibrated to any real organisation, so treat precision at this rate as indicative. Re-weight or subsample normal rows to test other prevalence levels.
- **Residual statistical signals.** Planted price spikes and inflated quantities raise line amounts (line amount alone separates classes with AUC ≈ 0.64). Historical spikes cannot occur in the first months because they need five prior purchases, though the overall monthly anomaly share stays between 8% and 15%. A model can partly learn these artifacts; prefer time-based splits and check feature importance.
- **Stylized prices and behavior.** Prices follow a smooth drift with bounded noise; no catalog changes, seasonal spikes, commodity shocks, volume discounts or supplier switching. Real price histories are noisier and the 1.30 threshold would need tuning.
- **Clean master data.** No typos, missing values, merged supplier records or unit-of-measure mismatches; real ERP extracts need substantial cleaning before any rule applies.
- **Exposure is an estimate.** Values follow the documented formulas; real recoverable amounts depend on contracts, credit notes and negotiation.
