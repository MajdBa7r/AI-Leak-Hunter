# Rule engine under realistic procurement noise

Rules only (Isolation Forest off), invoice-line level, labels = synthetic ground truth. Disruption intensities are those fixed in Phase 1 (`tests/test_robustness.py`), seed 7.

| Scenario | Precision | Recall | F1 | TP | FP | FN | Precision / recall excl. LOW-confidence | FP by finding type | FN by planted type |
|---|---|---|---|---|---|---|---|---|---|
| Clean data (no disruption) | 1.000 | 0.998 | 0.999 | 499 | 0 | 1 | 1.000 / 0.998 | - | {'QUANTITY_DISCREPANCY': 1} |
| Fuzzy duplicate invoice numbers (all 40 re-entries) | 1.000 | 0.992 | 0.996 | 496 | 0 | 4 | 1.000 / 0.992 | - | {'DUPLICATE_INVOICE': 3, 'QUANTITY_DISCREPANCY': 1} |
| 25% of re-entries booked under a duplicate vendor record | 1.000 | 0.978 | 0.989 | 489 | 0 | 11 | 1.000 / 0.978 | - | {'DUPLICATE_INVOICE': 10, 'QUANTITY_DISCREPANCY': 1} |
| Warehouse posting lag 21 days | 0.986 | 0.998 | 0.992 | 499 | 7 | 1 | 0.986 / 0.998 | {'QUANTITY_DISCREPANCY': 7} | {'QUANTITY_DISCREPANCY': 1} |
| Rounding noise (+/-1 cent, 3 decimals) on 10% of lines | 1.000 | 0.998 | 0.999 | 499 | 0 | 1 | 1.000 / 0.998 | - | {'QUANTITY_DISCREPANCY': 1} |
| Price amendments recorded 5 days after first invoice | 0.677 | 0.998 | 0.807 | 499 | 238 | 1 | 0.817 / 0.998 | {'PO_PRICE_MISMATCH': 238} | {'QUANTITY_DISCREPANCY': 1} |
| Legitimate 35% renegotiation (3 suppliers) | 0.915 | 0.996 | 0.954 | 498 | 46 | 2 | 0.915 / 0.996 | {'HISTORICAL_PRICE_SPIKE': 46} | {'PO_PRICE_MISMATCH': 1, 'QUANTITY_DISCREPANCY': 1} |
| Returns posted as negative receipts (5% of receipts) | 0.677 | 0.998 | 0.807 | 499 | 238 | 1 | 0.677 / 0.998 | {'QUANTITY_DISCREPANCY': 238} | {'QUANTITY_DISCREPANCY': 1} |
| ALL SEVEN COMBINED | 0.492 | 0.976 | 0.654 | 488 | 504 | 12 | 0.559 / 0.976 | {'PO_PRICE_MISMATCH': 235, 'QUANTITY_DISCREPANCY': 223, 'HISTORICAL_PRICE_SPIKE': 46} | {'DUPLICATE_INVOICE': 10, 'PO_PRICE_MISMATCH': 1, 'QUANTITY_DISCREPANCY': 1} |
