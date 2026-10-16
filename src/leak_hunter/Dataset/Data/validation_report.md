# AI Leak Hunter synthetic dataset: validation report

Seed 42. Activity window 2025-10-01 to 2026-09-30 (as-of date for payments).

## Validation checks

| # | Check | Result | Detail |
|---|---|---|---|
| 1 | Exactly 5,000 invoice lines | PASS | 5000 |
| 2 | Ground truth has one row per invoice line, same order | PASS | 5000 |
| 3 | 4,500 normal / 500 anomalous (exactly 10% anomaly rate) | PASS | normal=4500 anomalous=500 rate=10.00% |
| 4 | Primary anomaly types match the configured non-uniform mix | PASS | DUPLICATE_INVOICE=80 (16%), QUANTITY_DISCREPANCY=140 (28%), PO_PRICE_MISMATCH=170 (34%), HISTORICAL_PRICE_SPIKE=110 (22%) |
| 5 | is_anomaly consistent with primary_anomaly_type | PASS |  |
| 6 | invoice_line_id unique | PASS |  |
| 7 | supplier_id unique in suppliers | PASS |  |
| 8 | po_line_id unique in purchase_orders | PASS |  |
| 9 | receipt_id unique in goods_receipts | PASS |  |
| 10 | amendment_id unique in po_amendments | PASS |  |
| 11 | Supplier IDs follow SUP-0001 pattern, 80 suppliers | PASS |  |
| 12 | invoice_line_id is prefixed by its invoice_id | PASS |  |
| 13 | Each invoice document has one supplier, number, date, status and PO | PASS |  |
| 14 | Invoice supplier_id exists in suppliers | PASS |  |
| 15 | Invoice po_line_id exists in purchase_orders | PASS |  |
| 16 | Invoice item_id exists in the catalog | PASS |  |
| 17 | PO supplier_id exists in suppliers | PASS |  |
| 18 | Receipt po_line_id exists in purchase_orders | PASS |  |
| 19 | Amendment po_line_id exists in purchase_orders | PASS |  |
| 20 | PO approved_amendment_id resolves to an Approved amendment of the same PO line | PASS | 349 linked amendments |
| 21 | Approved amendments always referenced by their PO line | PASS |  |
| 22 | Invoice supplier matches PO supplier | PASS |  |
| 23 | Invoice item matches PO line item | PASS |  |
| 24 | Invoice po_id matches PO line's po_id | PASS |  |
| 25 | Receipt item and po_id match PO line | PASS |  |
| 26 | All quantities positive | PASS |  |
| 27 | All prices positive | PASS |  |
| 28 | Single currency USD | PASS |  |
| 29 | Monetary values have at most 2 decimals | PASS |  |
| 30 | line_amount = quantity_invoiced x unit_price | PASS |  |
| 31 | Receipt dates not before PO date | PASS |  |
| 32 | Invoice dates not before PO date | PASS |  |
| 33 | Invoice dates not before the PO's first delivery | PASS |  |
| 34 | Amendment dates between PO date and first invoice of the line | PASS |  |
| 35 | All dates inside the 12-month window | PASS | 2025-10-06 .. 2026-08-19 |
| 36 | Paid lines: amount_paid = line_amount | PASS |  |
| 37 | Unpaid lines: amount_paid = 0 | PASS |  |
| 38 | Partially Paid lines: 0 < amount_paid < line_amount | PASS |  |
| 39 | Flagged invoices are not all paid (all three statuses present among anomalies) | PASS | {'Paid': 389, 'Unpaid': 61, 'Partially Paid': 50} |
| 40 | DUPLICATE_INVOICE: every planted record fires its reference rule | PASS | 80/80 |
| 41 | QUANTITY_DISCREPANCY: every planted record fires its reference rule | PASS | 140/140 |
| 42 | PO_PRICE_MISMATCH: every planted record fires its reference rule | PASS | 170/170 |
| 43 | HISTORICAL_PRICE_SPIKE: every planted record fires its reference rule | PASS | 110/110 |
| 44 | No normal record fires any of the four rules | PASS | 0 violations |
| 45 | primary type contained in all_applicable_anomaly_types | PASS |  |
| 46 | Normal records have all_applicable = NORMAL and exposure 0 | PASS |  |
| 47 | Every line that fires a rule is a labelled anomaly | PASS |  |
| 48 | 40 duplicate pairs detectable by key (supplier_id, invoice_number) | PASS | 40 groups |
| 49 | No unplanned invoice-number repeats in the data | PASS |  |
| 50 | Duplicate pairs share PO line, item, quantity and price | PASS |  |
| 51 | Duplicate exposure counted once per pair | PASS |  |
| 52 | Duplicate pairs cover several payment-status combinations | PASS | Paid/Paid: 23; Unpaid/Paid: 4; Paid/Unpaid: 4; Paid/Partially Paid: 4; Partially Paid/Paid: 3; Partially Paid/Partially Paid: 1; Unpaid/Partially Paid: 1 |
| 53 | Quantity discrepancies persist after aggregating ALL receipts (incl. later ones) | PASS |  |
| 54 | Some quantity discrepancies involve multiple receipt lines | PASS | 24 cases |
| 55 | Quantity cases list their receipt IDs as evidence | PASS |  |
| 56 | PO price mismatches remain after applying approved amendments | PASS | 17 cases had an approved amendment applied first |
| 57 | Price mismatches include modest (<8%) and substantial (>=20%) overcharges | PASS | modest=76, moderate=67, substantial=27 |
| 58 | Historical spikes have >= 5 strictly-earlier lines for the same supplier/item | PASS | min prior lines = 5, median = 11 |
| 59 | Historical baselines use only past data (ground-truth median = rule median on prior lines) | PASS |  |
| 60 | Some historical spikes carry zero exposure (no contractual basis) | PASS | 77 zero-exposure, 33 with PO-based exposure |
| 61 | Normal partial deliveries later completed exist and are labelled normal | PASS | 289 lines |
| 62 | Normal lines priced above original PO via approved amendment exist and are not flagged | PASS | 264 lines |
| 63 | Normal lines with a rejected amendment on file are billed at the original price | PASS | 66 lines |
| 64 | Normal lines include 'Paid' invoices | PASS |  |
| 65 | Normal lines include 'Partially Paid' invoices | PASS |  |
| 66 | Normal lines include 'Unpaid' invoices | PASS |  |
| 67 | High-value lines are mixed normal/anomalous | PASS | 187 normal, 63 anomalous |
| 68 | Normal lines include multi-line invoices and multi-invoice POs | PASS |  |
| 69 | expected_exposure matches the documented formulas recomputed from the tables | PASS | 0 mismatches; total exposure $820,324.37 |
| 70 | No ground-truth or label-like columns in procurement tables | PASS | clean |
| 71 | No supplier/month/status/number-format/item value is exclusively anomalous (n>=10) | PASS | none |
| 72 | Anomalies spread across suppliers and months | PASS | 76 suppliers, 11 months |
| 73 | Anomalies interleaved through file order (every decile 3-20% anomalous) | PASS | 10% 9% 8% 9% 9% 11% 11% 11% 9% 12% |

**73/73 checks passed.**

## Table sizes

| File | Rows |
|---|---|
| suppliers.csv | 80 |
| purchase_orders.csv | 4,425 |
| po_amendments.csv | 433 |
| goods_receipts.csv | 5,541 |
| invoices.csv | 5,000 |
| ground_truth.csv | 5,000 |

2,218 purchase orders, 2,670 invoice documents, 80 distinct items invoiced across 8 categories, 80 suppliers. Supplier tiers: Small 35, Medium 31, Large 14.
PO amendments: PRICE Approved: 260, PRICE Rejected: 84, QUANTITY Approved: 89.
Ordinary price normalisations applied during generation: 0.

## Anomaly distribution

| Primary type | Records | Expected exposure | Mean | Max | Exposure paid to date |
|---|---|---|---|---|---|
| NORMAL | 4500 | $0.00 | $0.00 | $0.00 | $0.00 |
| DUPLICATE_INVOICE | 80 | $206,875.13 | $2,585.94 | $31,919.25 | $154,685.59 |
| QUANTITY_DISCREPANCY | 140 | $206,684.46 | $1,476.32 | $13,240.50 | $157,008.52 |
| PO_PRICE_MISMATCH | 170 | $167,642.37 | $986.13 | $28,994.10 | $137,501.18 |
| HISTORICAL_PRICE_SPIKE | 110 | $239,122.41 | $2,173.84 | $32,923.80 | $210,454.97 |
| **Total** | 5000 | $820,324.37 | | | $659,650.26 |

### All applicable labels (primary first)

| all_applicable_anomaly_types | Records |
|---|---|
| PO_PRICE_MISMATCH | 155 |
| QUANTITY_DISCREPANCY | 140 |
| HISTORICAL_PRICE_SPIKE | 77 |
| DUPLICATE_INVOICE | 40 |
| DUPLICATE_INVOICE|QUANTITY_DISCREPANCY | 40 |
| HISTORICAL_PRICE_SPIKE|PO_PRICE_MISMATCH | 33 |
| PO_PRICE_MISMATCH|HISTORICAL_PRICE_SPIKE | 15 |

### Planting variants

| Primary type | Variant | Records |
|---|---|---|
| DUPLICATE_INVOICE | DUP_FIRST_ENTRY | 40 |
| DUPLICATE_INVOICE | DUP_REPEAT_ENTRY | 40 |
| QUANTITY_DISCREPANCY | INVOICE_QTY_INFLATED | 84 |
| QUANTITY_DISCREPANCY | UNDELIVERED_BALANCE_BILLED | 56 |
| QUANTITY_DISCREPANCY | EXCESS_SMALL | 50 |
| QUANTITY_DISCREPANCY | EXCESS_MEDIUM | 60 |
| QUANTITY_DISCREPANCY | EXCESS_LARGE | 30 |
| PO_PRICE_MISMATCH | NO_AMENDMENT | 136 |
| PO_PRICE_MISMATCH | ABOVE_APPROVED_AMENDMENT | 17 |
| PO_PRICE_MISMATCH | REJECTED_AMENDMENT_PRICE_BILLED | 17 |
| PO_PRICE_MISMATCH | OVERCHARGE_MODEST | 76 |
| PO_PRICE_MISMATCH | OVERCHARGE_MODERATE | 67 |
| PO_PRICE_MISMATCH | OVERCHARGE_SUBSTANTIAL | 27 |
| HISTORICAL_PRICE_SPIKE | NO_CONTRACT_BASIS | 77 |
| HISTORICAL_PRICE_SPIKE | ALSO_ABOVE_PO_PRICE | 33 |
| HISTORICAL_PRICE_SPIKE | SPIKE_MODERATE | 33 |
| HISTORICAL_PRICE_SPIKE | SPIKE_HIGH | 48 |
| HISTORICAL_PRICE_SPIKE | SPIKE_EXTREME | 29 |

### Payment status by primary type

| Primary type | Paid | Partially Paid | Unpaid |
|---|---|---|---|
| NORMAL | 3647 | 385 | 468 |
| DUPLICATE_INVOICE | 61 | 10 | 9 |
| QUANTITY_DISCREPANCY | 107 | 12 | 21 |
| PO_PRICE_MISMATCH | 138 | 17 | 15 |
| HISTORICAL_PRICE_SPIKE | 83 | 11 | 16 |

### Normal scenario coverage

| Scenario tag | Normal records |
|---|---|
| APPROVED_PRICE_AMENDMENT_APPLIED | 264 |
| APPROVED_QTY_AMENDMENT | 95 |
| HIGH_VALUE | 187 |
| MULTIPLE_INVOICES_PER_PO | 1174 |
| MULTI_LINE_INVOICE | 3478 |
| MULTI_RECEIPT_PO_LINE | 1334 |
| PARTIAL_DELIVERY_LATER_COMPLETED | 289 |
| REJECTED_AMENDMENT_ON_FILE | 66 |
| VARIABLE_PRICE_SUPPLIER | 410 |

### Anomaly share by invoice month (leakage diagnostic)

| Month | Lines | Anomalous share |
|---|---|---|
| 2025-10 | 243 | 9% |
| 2025-11 | 519 | 10% |
| 2025-12 | 618 | 9% |
| 2026-01 | 511 | 8% |
| 2026-02 | 506 | 10% |
| 2026-03 | 525 | 10% |
| 2026-04 | 582 | 12% |
| 2026-05 | 581 | 11% |
| 2026-06 | 513 | 9% |
| 2026-07 | 355 | 13% |
| 2026-08 | 47 | 15% |

Historical-spike anomalies need at least five earlier purchases of the same item from the same supplier, so they cannot occur in the first months; the other types are spread across the year.

## File fingerprints (SHA-256, first 16 hex)

- suppliers.csv: `3e1893925abd23f7`
- purchase_orders.csv: `8dda3ed47520512f`
- po_amendments.csv: `1d384cad21695a25`
- goods_receipts.csv: `ef929c72d4a341a2`
- invoices.csv: `1cf225aaa9e26bb8`
- ground_truth.csv: `a6c055401144577d`
