# AI Leak Hunter

AI Leak Hunter is a procurement leakage investigation system. It reconciles purchase orders, goods receipts, price and quantity amendments, invoices and payments. It then flags suspicious invoice lines and groups them into cases, estimating the money at stake with documented formulas. Cases are ranked so investigators see the most valuable ones first, and every case comes with its evidence, calculations, next verification steps and an audit trail of the decision.

It is a decision-support tool. It separates four things that are often blurred:

| What the system reports | Meaning |
|---|---|
| **Statistical anomaly** | The line is unusual for this supplier or item. There is no contractual basis and no money is attached. |
| **Rule violation** | The line breaks a documented control (PO price, receipts, duplicate number). Its potential exposure is computed by formula. |
| **Paid exposure / confirmed loss** | Part of the exposure is already paid. A *confirmed* loss is claimed only when payment records alone prove it (a duplicate payment). |
| **Reviewer conclusion** | What a human decided: confirmed leakage, no issue, needs information, escalated. |

It does not claim to detect fraud.

---

## Quick start

```bash
pip install -r requirements-dev.txt                # app + pytest, matplotlib, shap (app only: requirements.txt)
bash scripts/run_all.sh                            # all outputs and reports, about 8 min
streamlit run app/streamlit_app.py                 # investigation app at http://localhost:8501
python -m pytest                                   # 125 tests + 2 documented known-gap xfails, about 6 min
```

On Windows (PowerShell), run the steps of `scripts/run_all.sh` one by one after `$env:PYTHONPATH="src"`. The minimum to use the app is:

```powershell
$env:PYTHONPATH="src"; python -m leak_hunter.pipeline --out-dir outputs; streamlit run app/streamlit_app.py
```

The app runs detection itself on first start if `outputs/` is empty, which is the case on a fresh Streamlit Cloud deploy (about 15 s). It needs no internet access: the copilot uses a model only if an API key is configured, and otherwise answers offline.

---

## The workflow

```
procurement CSVs ──► data_loader  validate schema, keys, relationships (never reads ground truth)
                     reconciliation  approved price at invoice date, approved qty, all receipts,
                                     cumulative invoicing per PO line, duplicate groups (exact + fuzzy)
                     history         strictly-earlier baselines per supplier and item
        ┌────────────┴─────────────┐
 detection_rules (A-D)       anomaly_detection (Isolation Forest, fitted on earliest 60%)
        └────────────┬─────────────┘
                findings  one schema, stable IDs, JSON evidence and details
                exposure  cents arithmetic; overlap-safe exposure per line and case
            prioritization  risk score + breakdown, expected value, bands, linked cases, supplier profiles
                     │
        outputs/  findings, incidents (cases), scores, pending checks, supplier profiles, run summary
                     │
 investigation  calculations, evidence records, evidence map, timeline, playbook, Markdown report
 case_store     append-only SQLite decision log (status, reviewer, note, recovered amount)
 app/           Streamlit + Plotly: overview, case queue, case file, suppliers, decision log, evaluation
```

| Component | Why it exists |
|---|---|
| Reconciliation with a grace period | Billing ahead of posted receipts is *pending*, not a violation, until the invoice is 30 days old. This absorbs warehouse posting lag (stress-tested). |
| Four rules with materiality | They catch control breaches an auditor can defend: duplicate (exact and fuzzy), quantity beyond receipts or the approved quantity, price above the approved price, and unusual price increases. Materiality thresholds (price > $5 and > 2%; quantity ≥ 1.5% and ≥ $25) stop pennies being flagged. |
| Overlap-safe exposure | One line can break several rules. Each line counts max(duplicate, quantity + max(price mismatch, price spike)), so cases never double count. |
| Isolation Forest | Ranks how unusual each line is, using point-in-time features only. It never assigns money. Its value is *queue precision under noise* (see Evaluation). |
| Risk score (0–100) | A transparent per-case urgency score: exposure (35) + evidence strength (30) + money already paid (15) + statistical anomaly (20). Every case shows the breakdown. |
| Expected value | Exposure × rule-confidence factor. This is the default queue order ("money first"), chosen by a pre-committed rule on an untouched replica dataset. |
| Linked cases and supplier profiles | Cases that share a PO line or invoice are linked, and supplier pages show where cases and exposure concentrate. These help navigation; they are not detectors. |
| Playbooks | Next verification steps generated from each case's evidence, e.g. "place a payment block on INV-x before the next payment run" when a repeat entry is still unpaid. |
| Decision log | An append-only audit trail (enforced by SQLite triggers). It also tracks the share of closed cases reviewers confirm as leakage, plus amounts recovered, so the queue's real-world precision can be measured once it is in use. |

---

## The app

- **Overview:** where the money stands (potential exposure, already paid, confirmed loss, awaiting receipts), cases by band, reviewer progress, exposure by case type and by month, and the top of the queue.
- **Case queue:** filter by band, type, supplier, status, evidence tier, minimum exposure and ID search. Sort order can be money first, risk score or severity label. Select a row to open the case file; the queue can be exported as CSV.
- **Case file:** a header with band and status; potential, paid and confirmed amounts; the risk-score breakdown.
  - *Why it was flagged:* each finding with its formula and actual numbers, plus the overlap-safe exposure table and linked cases.
  - *Evidence map:* a layered graph from supplier to PO lines, amendments and receipts, invoice lines and invoices, with duplicate links in red. A timeline shows what happened when.
  - *Records:* the source rows.
  - *Next steps:* a checklist from the playbook.
  - *Decision & report:* record an outcome; download a Markdown case report.
- **Suppliers:** spend vs exposure, per-supplier profile, and the supplier's cases.
- **Decision log:** every decision, with CSV export.
- **Evaluation:** every benchmark report, with a synthetic-data warning.

---

## Investigation copilot

The copilot is a chat on the **Overview** page (scope: the whole case queue) and in the **Case file > Copilot** tab (scope: the selected case). It offers three quick prompts plus free questions.

**Grounding.** `copilot.case_context` builds one JSON payload per case:
- the findings with their calculation steps and the de-duplicated exposure per line;
- the risk-score breakdown and the rule thresholds that applied;
- the supplier profile, linked cases and timeline;
- the source records and the playbook.

The model sees only that payload. The system prompt sets an auditor persona:
- cite IDs and amounts exactly;
- keep potential, paid and confirmed amounts distinct;
- never call anything fraud;
- answer "The evidence available to me does not show that." when the payload lacks the answer;
- write letters in a neutral, document-requesting tone.

The full payload is visible in the app under *Evidence the copilot can see*.

**Citation check.** After every answer, each transaction ID and dollar amount is looked up in the payload. The caption under the answer says either "all found in the evidence" or lists what was not found, so a hallucinated figure is visible immediately.

**Providers.** OpenAI, Groq or Google Gemini are called over HTTPS with `requests`; no SDKs are needed. The provider is chosen from environment variables or Streamlit secrets: the first key found among `OPENAI_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY`/`GOOGLE_API_KEY`, unless `LEAK_HUNTER_LLM_PROVIDER` says otherwise. Model names can be overridden with `OPENAI_MODEL`, `GROQ_MODEL` and `GEMINI_MODEL`.

**Offline copilot.** It runs with no key, with the sidebar switch *Copilot: offline mode only*, with `LEAK_HUNTER_COPILOT_OFFLINE=true`, or automatically whenever a provider call fails (timeout, network, quota, unknown model). Answers then come from deterministic templates over the same payload:
- risk summary with calculations;
- evidence, and what it does and doesn't prove;
- a neutral supplier inquiry letter;
- next steps;
- supplier profile, linked cases and timeline;
- rank explanation;
- a fraud-question refusal;
- portfolio questions.

The chat never errors during a demo.

### Streamlit Community Cloud

1. Push the repository, including `src/leak_hunter/Dataset/Data/*.csv`, `app/`, `.streamlit/config.toml` and `requirements.txt`.
2. New app, then set **Main file path** to `app/streamlit_app.py` and Python to 3.11 or newer.
3. Under **App settings > Secrets**, paste the following (one key is enough; with none, the copilot runs offline):

```toml
OPENAI_API_KEY = "sk-..."
# or: GROQ_API_KEY = "gsk_..."
# or: GEMINI_API_KEY = "AIza..."
# optional:
# LEAK_HUNTER_LLM_PROVIDER = "openai"   # openai | groq | gemini
# OPENAI_MODEL = "gpt-4o-mini"
# LEAK_HUNTER_COPILOT_OFFLINE = "true"  # force offline (booth safety)
```

All paths resolve from the repository root, and nothing machine-specific is required. Cloud storage is ephemeral, so `outputs/` (including the decision log) is rebuilt after each restart. For a persistent decision log, point `LEAK_HUNTER_CASE_DB` at durable storage. For local use, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`; it is git-ignored.

---

## Evaluation (synthetic data, out-of-time window)

All results are on synthetic data with planted anomalies. Reproduce them with `scripts/run_all.sh`.

### Rules (`reports_rules/`)

- **Clean data:** precision 1.00, recall 0.998; the one miss is an invoice younger than the grace period, shown as pending.
- **Seven realistic disruptions combined:** precision 0.49, recall 0.98. Most false alarms come from two gaps in the data, not the rules: price amendments recorded late, and returns with no credit note. Both need extra fields (Phase 2).

### Unsupervised detectors (`reports_ml/`)

- **Isolation Forest is best at the top of the queue:** the 10 highest-scored lines are all anomalies (P@10 = 1.00), and PR-AUC is 0.48.
- **Robust Mahalanobis ties on PR-AUC** and has the best ROC-AUC (0.91).
- **The fixed-weight ensemble came second to last**, because two weak members dilute it.
- **No ML recall gain over the rules:** on this data the models find nothing the rules miss.

### Investigation queues (`reports_queue/`)

This is the headline evaluation: what an investigator experiences. Metrics are on the out-of-time window.

| Scenario | Ordering | Precision @50 | Precision @100 | Exposure in first 50 cases |
|---|---|---|---|---|
| Original, clean | Rules by exposure / expected value | 1.00 | 1.00 | 84% |
| Original, clean | Integrated risk score | 1.00 | 1.00 | 77% |
| Original, 7 disruptions | Rules by exposure / expected value | 0.86 | 0.70 | 79% |
| Original, 7 disruptions | Integrated risk score | 0.88 | **0.81** | 69% |
| Original, 7 disruptions | Integrated without ML points | 0.82 | 0.61 | 74% |
| **Replica (seed 2026), 7 disruptions** | Expected value | 0.94 | 0.79 | **82%** |
| **Replica (seed 2026), 7 disruptions** | Integrated risk score | **0.98** | **0.92** | 74% |
| Any | ML only | 0.92–1.00 | 0.73–0.80 | 39–53% |

**What this establishes, on synthetic data:**

1. **Ordering by money captures the most exposure.** The pre-registered risk score log-scales money, so it reaches less of it in the first 50 cases.
2. **The ML layer earns its place through precision under noise, on both datasets.** Adding the Isolation Forest points raises precision in the top 100 cases from 0.61 to 0.81 on the original data, and from 0.79 to 0.92 on the replica.
3. **ML alone is a poor queue.** It finds unusual lines, not money.

**How the default order was chosen.** The risk-score weights were fixed before the first evaluation. When the risk score lost on exposure, an expected-value ordering was proposed. A decision rule was written down (`reports_queue/DECISION_RULE.md`) before testing it on a replica dataset that no component had seen. Expected value won, 82% vs 74% of exposure in the first 50 cases, so it is the default. The risk score remains the urgency band and an alternative sort. The original v1 results are kept in `reports_queue/original_dataset_v1/`.

### Robustness (`tests/test_robustness.py`)

Seven disruptions are codified as tests: fuzzy duplicate numbers, posting lag, rounding noise, renegotiation, returns, plus row-order and later-transaction invariance. Two known gaps are strict xfails, so they will fail loudly once Phase 2 closes them:
- a duplicate booked under a cloned vendor record;
- price amendments approved after the invoice but effective earlier.

---

## Limitations

- **Synthetic data only.** Every metric shows recovery of planted patterns. None of it is evidence of real-world fraud-detection performance, and the 10% anomaly rate is a property of the dataset.
- **Missing fields:**
  - No amendment effective date, so retroactive approvals become false alarms.
  - No credit notes or return movement types.
  - No supplier tax ID or bank account, so duplicate vendor records are missed.
  - No payment dates.
  - No approval thresholds, so split-purchase detection isn't possible.
- **Single runs.** The queue evaluation has one seeded run per scenario and no confidence intervals. The replica is generated by the same generator, so it tests a fresh draw, not a different world.
- **Judgement-based settings.** Risk-score weights, materiality thresholds and the 2% review budget are business judgement. They should be calibrated on real decision-log outcomes.
- **Isolation Forest limits.** It cannot extrapolate beyond its training range. Refit it periodically.

## Next steps

1. Dataset v2 (effective dates, credit notes, returns, vendor identifiers) to close the two strict-xfail gaps.
2. Once reviewers have recorded decisions, use them as labels to calibrate the risk-score weights on a held-out period (semi-supervised ranking).
3. Evaluate the online copilot on a fixed question set per case type: citation-check pass rate and refusal correctness. Only the offline copilot and mocked providers are tested here, because no live API was reachable from the build environment.
4. Approval-threshold and segregation-of-duties checks once the data includes approver and threshold fields.
