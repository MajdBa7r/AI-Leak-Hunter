# AI Leak Hunter: detection engine (stage 1)

> Product overview, app, prioritization and queue evaluation: see [PRODUCT.md](PRODUCT.md).

Rule-based leakage detection plus unsupervised anomaly scoring over the procurement tables in `src/leak_hunter/Dataset/Data/`. The engine reads only the five procurement CSVs. `ground_truth.csv` is read only by the evaluation step, after the findings have been written.

## Quick start

From the repository root:

```bash
pip install -r requirements.txt

# 1. Detect (no labels are read)
PYTHONPATH=src python -m leak_hunter.pipeline --out-dir outputs

# 2. Evaluate the saved findings against the synthetic ground truth
PYTHONPATH=src python -m leak_hunter.evaluation --out-dir outputs --report reports/detection_evaluation.md

# 3. Tests (unit + integration + robustness/stress suite, ~2.5 min)
python -m pytest
python -m pytest tests/test_robustness.py -rxX     # stress suite only, shows the known-gap xfails
```

On Windows PowerShell, set the path first with `$env:PYTHONPATH="src"`.

Useful options for `leak_hunter.pipeline`:

- `--data-dir` to point at another data folder. You can also set the `LEAK_HUNTER_DATA_DIR` environment variable.
- `--no-isolation-forest` to run the rules only.
- Isolation Forest settings: `--n-estimators`, `--random-state`, `--contamination`, `--fit-mode {temporal,all}`, `--train-fraction`, `--review-budget`.
- Rule settings: `--as-of YYYY-MM-DD` (the extraction date used to compute invoice ages; defaults to the latest invoice or receipt date in the data), `--qty-grace-days`, `--no-fuzzy-duplicates`.

Every default lives in `config.py`.

## Architecture

```
data_loader.py      load + validate the 5 procurement CSVs (clear errors; rejects label columns)
      │
reconciliation.py   one row per invoice line: approved price at invoice date, approved quantity,
      │             all receipts aggregated, cumulative invoicing per PO line, duplicate groups
      ├── history.py            point-in-time baselines (strictly earlier invoice dates only)
      │
      ├── detection_rules.py    Rules A-D → hit tables
      └── anomaly_detection.py  features → Isolation Forest (fit on training window) → scores, hits
                │
findings.py         common findings schema, stable IDs, incident grouping, severity, validation
exposure.py         cents arithmetic, payment proration, duplicate-payment logic, de-duplicated incident exposure
pipeline.py         CLI: run everything, write outputs/ (never reads ground truth)
evaluation.py       CLI: read outputs/, then ground truth → metrics + reports/detection_evaluation.md
```

The repository's existing secret scanner (`main.py`, `scanner.py`, `detector.py`, `rules.py`) is untouched. The new modules use different names so nothing collides.

## Rules

| Rule | Flags | Potential exposure |
|---|---|---|
| A: Duplicate invoice | Documents from the same supplier that are entries of the same invoice. **Exact match:** numbers equal after normalisation (upper-case, letters and digits only), any amount. **Fuzzy match:** identical document amount, invoice dates ≤ 60 days apart, and numbers equal after mapping look-alike letters (O/Q→0, I/L→1) and ignoring leading zeros, or within one edit (insert, delete, substitute, adjacent swap). A single digit-for-digit change (INV-1041 vs INV-1042) is *not* matched, because that is how consecutive recurring invoices are numbered; Rule B covers that case when it over-bills. The earliest entry is the original; later ones are repeats. Both entries are reported and share one incident. | Repeat line: its line amount. Original line: 0. Confirmed loss = max(0, paid on both − original amount), i.e. only when payments prove a duplicate payment. |
| B: Quantity discrepancy | Per PO line, cumulative invoiced quantity in date order exceeds `min(sum of ALL receipts, approved quantity)`. The excess is attributed to the line that crosses the limit. **Grace period (30 days):** for younger invoices only the units beyond the *approved* quantity are flagged; units still awaiting receipt go to `pending_quantity_checks.csv` (PENDING_RECEIPT) and become findings once the invoice is older than the grace period. | flagged excess units × approved unit price |
| C: PO price mismatch | Invoice price exceeds the approved price by more than **max($0.01, 0.5%) per unit** (rounding tolerance). The approved price is the PO price, or the latest approved price amendment dated on or before the invoice. Rejected and pending amendments are ignored. | (invoice − approved) × quantity invoiced: the full difference, not just the part above the tolerance |
| D: Historical price spike | Price > max(1.30, 1 + 4 × robust CV) × median of ≥ 5 valid, strictly earlier prices for the same supplier and item. Duplicate re-entries are not counted as history. | Only the contractual excess over the approved price, above the Rule C tolerance. 0 when the PO itself carries the high price, because an unusual price is not a proven loss. |

**Paid exposure** for rules B–D is the exposure × amount_paid / line_amount. **Confirmed loss** is non-zero only for realized duplicate payments.

**De-duplication** works per invoice line:

- price component = max(C, D)
- line exposure = max(duplicate, quantity + price component)

Incident exposure is the sum over the incident's lines. Totals should always come from `incidents.csv`, never from summing `findings.csv`.

## Isolation Forest

**Features** (no IDs, no names, no labels):

- line amount
- price vs approved price
- share billed ahead of receipt: `max(0, cumulative invoiced − received by the invoice date) / cumulative invoiced`. It only counts over-billing, and only receipts dated on or before the invoice are visible.
- price vs earlier median, as a ratio and as a robust z-score
- whether that price history exists
- amount vs the supplier's earlier median amount
- supplier invoice count over the prior 30 days
- days from PO to invoice

**Leakage controls:**

- Historical features use earlier dates only.
- In temporal mode, the imputer, the model and the review threshold are fitted on the earliest 60% of lines.
- `contamination="auto"`. The dataset's 10% planted rate is not used.

**Outputs:**

- `anomaly_score`: higher means more unusual. It is a ranking score, not a probability.
- `risk_rank`.
- `top_features`: robust z-scores against the training window, showing which features drove the score.
- An `UNUSUAL_TRANSACTION` finding for each line above the review threshold. If the line already has a rule finding, it joins that incident instead of opening a new one.

## Findings schema (`outputs/findings.csv`, `findings.jsonl`)

| Column | Meaning |
|---|---|
| `finding_id` | Unique ID, stable across runs: `F-<DUP\|QTY\|PRC\|HPS\|IFR>-<invoice_line_id>` |
| `incident_id` | `INC-DUP-<supplier>-<number>` for duplicate groups, otherwise `INC-<invoice_line_id>` |
| `finding_type` | One of the four rule types or `UNUSUAL_TRANSACTION` |
| `detection_method` | `RULE_BASED` or `ISOLATION_FOREST` |
| `rule_id` | Which rule or model raised the finding |
| `invoice_id`, `invoice_line_id`, `invoice_number`, `supplier_id`, `po_id`, `po_line_id`, `item_id`, `invoice_date` | Context for the invoice line |
| `severity` | `CRITICAL` (confirmed duplicate payment), `HIGH`, `MEDIUM` or `LOW`, set by type and exposure |
| `potential_exposure`, `paid_exposure`, `confirmed_loss` | USD, two decimals; kept separate (see Rules) |
| `amount_paid`, `line_amount`, `currency` | Payment context for the line |
| `exposure_basis` | Formula used for the exposure |
| `evidence_record_ids` | JSON: `{"invoice_lines": [...], "invoices": [...], "po_lines": [...], "receipts": [...], "amendments": [...], "suppliers": [...]}` |
| `details` | JSON with the observed and reference values: prices, quantities, baselines, payment assessment, model drivers |
| `anomaly_score`, `risk_rank` | Isolation Forest findings only |
| `related_finding_ids` | Other findings in the same incident |
| `description` | Plain-language explanation |
| `status` | `OPEN` (later `UNDER_REVIEW`, `CONFIRMED`, `DISMISSED`) |

Other outputs:

- `incidents.csv`: one row per incident, with priority rank, de-duplicated exposure, maximum severity and finding IDs.
- `if_scores.csv`: score, rank and features for every line.
- `line_reconciliation.csv`: the reconciliation values for each line.
- `pending_quantity_checks.csv`: lines billed beyond posted receipts that are still inside the grace period. For each: days until it would be flagged, pending units, and the amount at stake if the goods never arrive.
- `run_summary.json`: configuration, data-quality issues and counts.

## Robustness suite (`tests/test_robustness.py`)

Each test applies one realistic disruption to a copy of the repository data and checks the engine's behaviour. These are the results on this dataset.

| Disruption | v1 | Phase 1 (hardened) | Test asserts |
|---|---|---|---|
| Repeat invoice numbers: suffix, swap, O for 0, dropped or added leading zero, case and spacing | 0–40 of 40 pairs found | **40/40 pairs, 0 false alarms**, for each variant | all pairs found, no false alarms |
| Repeat booked under a duplicate vendor ID | 0/40 | 0/40 (**known gap**) | strict xfail; partial safety net: the repeats are still caught by Rule B |
| Receipts posted 21 days late | 35 false alarms | **7** (all billed ahead of a delivery that was not yet posted) | ≤ 25% of the no-grace count, and only on PO lines that lost receipts |
| Receipts posted 60 days late (longer than the grace period) | 428 | 330 | no missed anomalies, and alarms only on PO lines that lost receipts |
| ±1 cent / 3-decimal price noise on 10% of lines | 214 false alarms | **0** | 0 false alarms with the default tolerance (and > 100 with zero tolerance) |
| Price amendments approved after the invoice but effective before it | 240 | 240 (**known gap**) | strict xfail: the schema has no `effective_date` |
| Legitimate 35% renegotiation on new POs | spike flags | spike flags only, exposure $0, LOW severity | no monetary rule fires |
| Returns as negative receipts | 255 | flags confined to returned quantities | flags only on PO lines with returns and never above the returned quantity; the loader reports the negative receipts |
| Row order shuffled; later transactions added | n/a | identical findings | metamorphic invariance |

Two tests are marked strict xfail: the vendor-record gap and the retroactive-amendment gap. When Phase 2 closes them they will report XPASS and fail, so the marker has to be removed at that point.

## Next stages

- **Evidence graph (NetworkX).** `evidence_record_ids` already lists typed node IDs. Build nodes for suppliers, POs, PO lines, receipts, amendments, invoices and invoice lines from the CSVs, add an edge from each finding to every evidence ID, and group nodes by `incident_id`. Duplicate pairs and shared PO lines then become connected components to explore.
- **AI investigation copilot.** Each finding's `details` and `description` give a grounded, structured context: observed values, reference values, formulas, payment status. A copilot can explain a case, draft supplier queries or recovery letters, and must cite the `evidence_record_ids` rather than invent figures. Status updates (`UNDER_REVIEW`, `CONFIRMED`, `DISMISSED`) feed back into `incidents.csv`.
- **Streamlit dashboard.**
  - KPI tiles from `incidents.csv`: de-duplicated potential exposure, paid exposure and confirmed loss.
  - An incident queue sorted by `priority_rank`.
  - Filters by type, method, severity and supplier.
  - A detail panel that parses the `details` and `evidence` JSON, with the price-history series from `prior_unit_prices` and the receipts timeline from Rule B.
