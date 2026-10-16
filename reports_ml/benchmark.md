# Unsupervised anomaly detection benchmark - AI Leak Hunter

Out-of-time evaluation: fitted on the earliest 60% of invoice lines (2,980), scored on the remaining 2,020 (planted-anomaly base rate 10.8%). 13 leak-free features; no labels used for fitting, tuning, weighting or thresholds.

| # | Model | Family | P@10 | P@25 | P@50 | ROC-AUC | PR-AUC (95% CI) | FPR @ 50% recall | FPR @ 80% recall | Review-budget precision / recall | Latency (ms / 1,000 txns) | Fit (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Isolation Forest** | tree | 1.00 | 0.80 | 0.70 | 0.841 | **0.476** (0.411-0.546) | 0.069 | 0.286 | 0.64 / 0.22 (73 flagged) | 57.3 | 0.8 |
| 2a | Local Outlier Factor | distance | 0.90 | 0.52 | 0.38 | 0.556 | 0.208 (0.162-0.268) | 0.380 | 0.899 | 0.32 / 0.14 (98 flagged) | 42.1 | 0.1 |
| 2b | ECOD | distribution | 0.80 | 0.48 | 0.34 | 0.770 | 0.317 (0.266-0.382) | 0.110 | 0.398 | 0.37 / 0.15 (87 flagged) | 2.4 | 0.0 |
| 3 | Robust Mahalanobis (MCD) | linear / covariance | 0.70 | 0.36 | 0.32 | 0.913 | 0.475 (0.425-0.541) | 0.061 | 0.064 | 0.32 / 0.04 (28 flagged) | 0.4 | 1.8 |
| 4 | Autoencoder (MLP 32-8-32) | neural reconstruction | 0.30 | 0.32 | 0.30 | 0.719 | 0.263 (0.218-0.318) | 0.128 | 0.567 | 0.34 / 0.20 (129 flagged) | 0.8 | 8.2 |
| 5 | Consensus ensemble (IF + LOF + AE) | rank average | 0.60 | 0.52 | 0.42 | 0.708 | 0.237 (0.194-0.299) | 0.239 | 0.450 | 0.32 / 0.19 (127 flagged) | 102.6 | 9.4 |

Review budget = lines scoring above the 98th percentile of training scores.

Planted anomalies flagged at the review budget, by type:

| Model | Price mismatch | Quantity | Price spike | Duplicate |
|---|---|---|---|---|
| Isolation Forest | 14/64 | 1/62 | 32/65 | 0/27 |
| Local Outlier Factor | 21/64 | 3/62 | 6/65 | 1/27 |
| ECOD | 12/64 | 4/62 | 16/65 | 0/27 |
| Robust Mahalanobis (MCD) | 1/64 | 2/62 | 6/65 | 0/27 |
| Autoencoder (MLP 32-8-32) | 9/64 | 8/62 | 27/65 | 0/27 |
| Consensus ensemble (IF + LOF + AE) | 20/64 | 10/62 | 10/65 | 1/27 |

#1 transaction of the winner (Isolation Forest): `INV-002273-02` - ground truth HISTORICAL_PRICE_SPIKE. Top attributions: price_vs_approved_rel +2.124, hist_price_robust_z +1.456, hist_price_log_ratio +1.363, log_qty_vs_prior_median_same_item +0.764. Autoencoder error share: price_vs_approved_rel 73%, log_amount_vs_supplier_median 9%, log_line_amount 7%.

