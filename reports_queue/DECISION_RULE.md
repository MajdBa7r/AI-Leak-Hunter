# Pre-committed decision rule for the default queue order

Written 2026-10-16T13:05:57+03:00, BEFORE any ordering was evaluated on the replica dataset.

Context: on the original dataset the pre-registered INTEGRATED risk score (v1 weights, unchanged)
captured less exposure in the first 50 cases than ordering rule incidents by exposure
(clean 77% vs 84%; all disruptions combined 69% vs 79%). Diagnosis: log-scaled exposure points
compress large and small cases, so non-monetary points (evidence, payment, anomaly) reorder
high-value cases below low-value ones.

Candidate added: EXPECTED_VALUE = de-duplicated potential exposure x confidence factor
(HIGH 1.0, MEDIUM 0.6, LOW 0.25; no contractual basis -> 0), ties and zero-value cases ordered
by the v1 risk score. Factors are business judgement, not fitted.

Because this candidate was proposed AFTER seeing original-dataset results, it is validated on a
fresh replica generated with the same generator and seed 2026 (73/73 generator checks passed),
which no detector, weight or threshold has seen.

Rule: on the replica, scenario "ALL SEVEN COMBINED", pick the default queue order with the higher
exposure captured @50 cases; if the two are within 2 percentage points, pick the one with the
higher precision @50. Candidates: INTEGRATED (v1 risk score) vs EXPECTED_VALUE. Report both
regardless of outcome; the original-dataset results remain published unchanged.

## Outcome (2026-10-16T13:06:55+03:00)

Replica (seed 2026), ALL SEVEN COMBINED, first 50 cases:

| Ordering | Exposure captured @50 | Precision @50 |
|---|---|---|
| INTEGRATED (v1 risk score) | 74% | 0.98 |
| EXPECTED_VALUE | 82% | 0.94 |
| Rules by exposure (reference) | 82% | 0.94 |

Difference 8 points > 2 -> default queue order = EXPECTED_VALUE (config.queue_order).
The risk score is kept as the per-case urgency band and explanation; it trades ~8 points of
exposure capture for ~4 points of precision under noise, so the app offers it as an alternative sort.
Expected value adds little over plain exposure ordering here because almost all rule findings are
HIGH confidence; it differs only for LOW/MEDIUM-confidence cases (e.g. borderline amendment timing).
