# AI Leak Hunter - detection evaluation

Generated 2026-10-16 09:17 UTC by `python -m leak_hunter.evaluation` from saved detection outputs. Python 3.13.16, pandas 3.0.5, numpy 2.5.3, scikit-learn 1.9.1.

> **Synthetic data.** These results measure how well the engine recovers anomalies that were planted with known rules in a synthetic dataset. They do **not** establish real-world fraud or leakage detection performance. The 10% anomaly rate is a property of this dataset, not an estimate of real prevalence.

## 1. Dataset and run

- Invoice lines: **5,000** (4,500 normal, 500 planted anomalies = 10%); 40 duplicate pairs.
- Planted primary types: PO_PRICE_MISMATCH 170, QUANTITY_DISCREPANCY 140, HISTORICAL_PRICE_SPIKE 110, DUPLICATE_INVOICE 80.
- Inputs: suppliers 80, purchase_orders 4,425, po_amendments 433, goods_receipts 5,541, invoices 5,000. Data-quality issues found by the loader: 0.
- Detection read ground truth: **False**. Labels were loaded only for this report.
- Findings: 587 (PO_PRICE_MISMATCH 203, QUANTITY_DISCREPANCY 179, HISTORICAL_PRICE_SPIKE 125, DUPLICATE_INVOICE 80); incidents: 459. Detection runtime 7.64 s.

## 2. Rule-based detection

Unit: **invoice line**. A line is predicted anomalous if any rule finding exists.

| Metric | Value |
|---|---|
| Precision | 1.0000 |
| Recall | 0.9980 |
| F1 | 0.9990 |
| TP / FP / FN / TN | 499 / 0 / 1 / 4500 |

Confusion matrix (lines):

| | Predicted anomalous | Predicted normal |
|---|---|---|
| **Anomalous (truth)** | 499 | 1 |
| **Normal (truth)** | 0 | 4500 |

### Per anomaly type (multi-label)

Truth = type listed in `all_applicable_anomaly_types` (a line can carry several types, e.g. a duplicate repeat entry also over-bills the received quantity); prediction = the line has a finding of that type.

| Type | Support | Precision | Recall | F1 | FP | FN |
|---|---|---|---|---|---|---|
| DUPLICATE_INVOICE | 80 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| QUANTITY_DISCREPANCY | 180 | 1.0000 | 0.9944 | 0.9972 | 0 | 1 |
| PO_PRICE_MISMATCH | 203 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| HISTORICAL_PRICE_SPIKE | 125 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |

Recall by planted primary type:

| Primary type | Planted | Found by its own rule | Found by any rule |
|---|---|---|---|
| DUPLICATE_INVOICE | 80 | 80 | 80 |
| QUANTITY_DISCREPANCY | 140 | 139 | 139 |
| PO_PRICE_MISMATCH | 170 | 170 | 170 |
| HISTORICAL_PRICE_SPIKE | 110 | 110 | 110 |

Ground-truth primary type (rows) vs finding types raised (columns; a line can appear in several columns):

| Truth \ finding | DUPLICATE_INVOICE | QUANTITY_DISCREPANCY | PO_PRICE_MISMATCH | HISTORICAL_PRICE_SPIKE | NO_RULE_FINDING | Lines |
|---|---|---|---|---|---|---|
| NORMAL | 0 | 0 | 0 | 0 | 4500 | 4500 |
| DUPLICATE_INVOICE | 80 | 40 | 0 | 0 | 0 | 80 |
| QUANTITY_DISCREPANCY | 0 | 139 | 0 | 0 | 1 | 140 |
| PO_PRICE_MISMATCH | 0 | 0 | 170 | 15 | 0 | 170 |
| HISTORICAL_PRICE_SPIKE | 0 | 0 | 33 | 110 | 0 | 110 |

Lines whose set of finding types equals the ground-truth type set exactly: 4,999 / 5,000.

### Other evaluation units

| Unit | Precision | Recall | Notes |
|---|---|---|---|
| Invoice document | 1.0000 | 1.0000 | document anomalous if any line is |
| Incident | 1.0000 | 0.9978 | duplicate pair = 1 incident; 460 true vs 459 predicted rule incidents, exact line-set match |
| Duplicate pair | - | - | 40/40 true pairs matched; 40/40 predicted groups are true pairs |

### False positives and false negatives

- False positives among normal lines: **0** 
- False negatives among planted anomalies: **1** {'QUANTITY_DISCREPANCY': 1}, of which 1 are PENDING: billed beyond posted receipts but younger than the 30-day posting-lag grace period, so deferred rather than cleared (they are flagged automatically once older). Recall counting pending lines as deferred: 1.0000. Normal lines pending: 0.

False-negative lines (first 20):

| invoice_line_id | primary type | status |
|---|---|---|
| INV-002644-03 | QUANTITY_DISCREPANCY | PENDING (grace period) |

## 3. Exposure

Compared on the 499 true-positive lines using the de-duplicated line exposure (duplicate pair counted once; price-rule overlap counted once).

| Item | Value |
|---|---|
| Lines whose exposure matches ground truth to the cent | 499 / 499 |
| Mean absolute difference per line | $0.0000 |
| Predicted potential exposure (de-duplicated) | $819,600.65 |
| Ground-truth expected exposure | $820,324.37 |
| Naive sum of rule findings (double counts overlaps; do not use) | $1,328,768.30 |
| Exposure attached to false-positive lines | $0.00 |
| Paid exposure: predicted vs ground truth | $659,650.26 vs $659,650.26 (499 exact line matches) |
| Confirmed loss (realized duplicate payments only) | $154,685.59 |

| Primary type | Lines | Exact | Predicted | Expected | Max abs diff |
|---|---|---|---|---|---|
| DUPLICATE_INVOICE | 80 | 80 | $206,875.13 | $206,875.13 | $0.00 |
| QUANTITY_DISCREPANCY | 139 | 139 | $205,960.74 | $205,960.74 | $0.00 |
| PO_PRICE_MISMATCH | 170 | 170 | $167,642.37 | $167,642.37 | $0.00 |
| HISTORICAL_PRICE_SPIKE | 110 | 110 | $239,122.41 | $239,122.41 | $0.00 |

## 5. Configuration

```json
{
  "rules": {
    "invoice_number_strip_leading_zeros": false,
    "dup_fuzzy": true,
    "dup_fuzzy_max_distance": 2,
    "dup_fuzzy_min_length": 5,
    "dup_fuzzy_amount_tolerance": 0.0,
    "dup_fuzzy_window_days": 60,
    "dup_fuzzy_allow_digit_substitution": false,
    "dup_fuzzy_require_same_items": true,
    "dup_recurring_interval_days": 30,
    "qty_tolerance_units": 0,
    "qty_materiality_pct": 0.015,
    "qty_materiality_value": 25.0,
    "qty_grace_days": 30,
    "as_of_date": null,
    "price_tolerance_abs": 0.01,
    "price_tolerance_pct": 0.005,
    "price_materiality_abs": 5.0,
    "price_materiality_pct": 0.02,
    "amendment_borderline_business_days": 3,
    "pending_amendment_statuses": [
      "pending",
      "draft",
      "submitted",
      "in review",
      "in_review"
    ],
    "hist_min_prior": 5,
    "hist_baseline": "median",
    "hist_ratio_threshold": 1.3,
    "hist_adaptive": true,
    "hist_dispersion_k": 4.0,
    "hist_exclude_duplicate_reentries": true
  },
  "isolation_forest": {
    "enabled": false,
    "n_estimators": 300,
    "max_samples": "auto",
    "max_features": 1.0,
    "contamination": "auto",
    "random_state": 42,
    "fit_mode": "temporal",
    "train_fraction": 0.6,
    "review_budget": 0.02,
    "supplier_history_min_prior": 5,
    "frequency_window_days": 30,
    "explain_top_n": 3
  },
  "severity": {
    "high_exposure": 5000.0,
    "medium_exposure": 500.0,
    "if_high_percentile": 0.995
  },
  "currency": "USD"
}
```

Rule settings follow the dataset's documented rule definitions (historical baseline: median of >= 5 strictly earlier observations, threshold 1.3x, adaptive widening k = 4.0); none were tuned on the labels.

## 6. Interpretation and limitations

- **Perfect or near-perfect rule scores are expected here and are not evidence of real-world accuracy.** The anomalies were planted with the same definitions the rules implement (the rules were written from the documented definitions, not tuned on labels), data are clean, invoice numbers repeat exactly, and there are no returns, credit notes, unit-of-measure issues or currency effects.
- Isolation Forest was disabled for this run.
- **Isolation Forest cannot extrapolate.** Split points are drawn inside the training range, so in temporal mode a value more extreme than anything in the training window scores like the most extreme training value, and features that were constant in training carry no signal. Refit the model regularly and keep the deterministic rules as the primary control.
- **Timing.** Rule B reconciles against all receipts posted by the as-of date, with a 30-day grace period for recent invoices (as-of date 2026-08-19); the Isolation Forest's quantity feature uses only receipts dated on or before the invoice. Lags longer than the grace period still produce quantity alarms (see tests/test_robustness.py).
- **Known gaps (tests/test_robustness.py, strict xfail):** a duplicate booked under a second vendor record, and price amendments approved after the invoice but effective before it, are not handled: the schema has no vendor tax/bank identifiers and no amendment effective date (Phase 2).
- **Exposure is potential, not proven.** Only realized duplicate payments are reported as confirmed loss; price and quantity overpayments need investigation (credit notes, late receipts) before they are losses.
- **Prevalence.** Precision depends on the anomaly base rate; at real-world rates far below 10%, the same detectors would show lower precision.
