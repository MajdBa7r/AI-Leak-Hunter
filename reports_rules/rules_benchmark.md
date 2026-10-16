# Rule engine benchmark - AI Leak Hunter

Rules only (Isolation Forest not used), unit = invoice line (5,000 lines: 4,500 normal, 500 planted anomalies). Per-class rows are multi-label (a line can carry several types).

### A. Clean synthetic data

| | Predicted anomaly | Predicted normal |
|---|---|---|
| **Actual anomaly** | 499 | 1 |
| **Actual normal** | 0 | 4500 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| DUPLICATE_INVOICE | 1.000 | 1.000 | 1.000 | 80 |
| QUANTITY_DISCREPANCY | 1.000 | 0.994 | 0.997 | 180 |
| PO_PRICE_MISMATCH | 1.000 | 1.000 | 1.000 | 203 |
| HISTORICAL_PRICE_SPIKE | 1.000 | 1.000 | 1.000 | 125 |
| **Any anomaly (line level)** | **1.000** | **0.998** | **0.999** | 500 |

Accuracy 1.000 (dominated by the 90% normal class; use precision/recall).

### B. Same data with all seven realistic disruptions combined (Phase 1 intensities)

| | Predicted anomaly | Predicted normal |
|---|---|---|
| **Actual anomaly** | 488 | 12 |
| **Actual normal** | 504 | 3996 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| DUPLICATE_INVOICE | 1.000 | 0.725 | 0.841 | 80 |
| QUANTITY_DISCREPANCY | 0.407 | 0.994 | 0.577 | 180 |
| PO_PRICE_MISMATCH | 0.453 | 0.995 | 0.622 | 203 |
| HISTORICAL_PRICE_SPIKE | 0.731 | 1.000 | 0.845 | 125 |
| **Any anomaly (line level)** | **0.492** | **0.976** | **0.654** | 500 |

Accuracy 0.897 (dominated by the 90% normal class; use precision/recall).
