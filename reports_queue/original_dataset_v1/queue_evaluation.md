# Investigation-queue evaluation

Unit = case (an incident; for ML-only, a single invoice line). Window = invoice lines after the Isolation Forest training cutoff (out-of-time). Labels are used only to score the queues. Risk-score weights were fixed in `config.PrioritizationConfig` before this evaluation was first run and have not been changed since. Single run per scenario (seeded); no confidence intervals.

## Clean data (no disruption)

| Ordering | Queue | P@25 | P@50 | P@100 | Exposure @25 | @50 | @100 | @200 | Cases to 80% of exposure | Queue precision | Line recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rules, by severity label (previous default) | 204 | 1.00 | 1.00 | 1.00 | 63% | 83% | 96% | 100% | 44 | 1.00 | 1.00 |
| Rules, by exposure | 204 | 1.00 | 1.00 | 1.00 | 67% | 84% | 96% | 100% | 43 | 1.00 | 1.00 |
| ML only (Isolation Forest score) | 2020 | 1.00 | 1.00 | 0.80 | 42% | 46% | 55% | 68% | 328 | 0.11 | 1.00 |
| Integrated risk score (rules + ML) | 213 | 1.00 | 1.00 | 1.00 | 54% | 77% | 91% | 100% | 67 | 0.96 | 1.00 |
| Integrated risk score without ML points | 213 | 1.00 | 1.00 | 1.00 | 63% | 77% | 93% | 100% | 54 | 0.96 | 1.00 |

## Fuzzy duplicate invoice numbers (all 40 re-entries)

| Ordering | Queue | P@25 | P@50 | P@100 | Exposure @25 | @50 | @100 | @200 | Cases to 80% of exposure | Queue precision | Line recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rules, by severity label (previous default) | 204 | 1.00 | 1.00 | 1.00 | 63% | 83% | 96% | 100% | 44 | 1.00 | 0.99 |
| Rules, by exposure | 204 | 1.00 | 1.00 | 1.00 | 67% | 84% | 96% | 100% | 43 | 1.00 | 0.99 |
| ML only (Isolation Forest score) | 2020 | 1.00 | 1.00 | 0.81 | 42% | 46% | 55% | 63% | 328 | 0.11 | 1.00 |
| Integrated risk score (rules + ML) | 212 | 1.00 | 1.00 | 1.00 | 54% | 76% | 91% | 100% | 70 | 0.96 | 0.99 |
| Integrated risk score without ML points | 212 | 1.00 | 1.00 | 1.00 | 63% | 77% | 93% | 100% | 54 | 0.96 | 0.99 |

## 25% of re-entries booked under a duplicate vendor record

| Ordering | Queue | P@25 | P@50 | P@100 | Exposure @25 | @50 | @100 | @200 | Cases to 80% of exposure | Queue precision | Line recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rules, by severity label (previous default) | 204 | 1.00 | 1.00 | 1.00 | 64% | 83% | 96% | 100% | 44 | 1.00 | 0.98 |
| Rules, by exposure | 204 | 1.00 | 1.00 | 1.00 | 67% | 84% | 96% | 100% | 43 | 1.00 | 0.98 |
| ML only (Isolation Forest score) | 2020 | 1.00 | 1.00 | 0.80 | 41% | 46% | 55% | 64% | 310 | 0.11 | 1.00 |
| Integrated risk score (rules + ML) | 214 | 1.00 | 1.00 | 1.00 | 55% | 74% | 93% | 100% | 59 | 0.95 | 0.98 |
| Integrated risk score without ML points | 214 | 1.00 | 1.00 | 1.00 | 64% | 78% | 93% | 100% | 55 | 0.95 | 0.98 |

## Warehouse posting lag 21 days

| Ordering | Queue | P@25 | P@50 | P@100 | Exposure @25 | @50 | @100 | @200 | Cases to 80% of exposure | Queue precision | Line recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rules, by severity label (previous default) | 211 | 1.00 | 0.90 | 0.93 | 64% | 81% | 96% | 100% | 47 | 0.97 | 1.00 |
| Rules, by exposure | 211 | 0.96 | 0.90 | 0.93 | 65% | 81% | 96% | 100% | 48 | 0.97 | 1.00 |
| ML only (Isolation Forest score) | 2020 | 1.00 | 0.94 | 0.77 | 42% | 46% | 55% | 63% | 341 | 0.11 | 1.00 |
| Integrated risk score (rules + ML) | 234 | 1.00 | 0.92 | 0.93 | 54% | 74% | 90% | 100% | 71 | 0.88 | 1.00 |
| Integrated risk score without ML points | 234 | 0.96 | 0.88 | 0.93 | 62% | 75% | 91% | 100% | 61 | 0.88 | 1.00 |

## Rounding noise (+/-1 cent, 3 decimals) on 10% of lines

| Ordering | Queue | P@25 | P@50 | P@100 | Exposure @25 | @50 | @100 | @200 | Cases to 80% of exposure | Queue precision | Line recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rules, by severity label (previous default) | 204 | 1.00 | 1.00 | 1.00 | 63% | 83% | 96% | 100% | 44 | 1.00 | 1.00 |
| Rules, by exposure | 204 | 1.00 | 1.00 | 1.00 | 67% | 84% | 96% | 100% | 43 | 1.00 | 1.00 |
| ML only (Isolation Forest score) | 2020 | 1.00 | 1.00 | 0.84 | 42% | 49% | 56% | 64% | 309 | 0.11 | 1.00 |
| Integrated risk score (rules + ML) | 209 | 1.00 | 1.00 | 1.00 | 55% | 76% | 91% | 100% | 62 | 0.98 | 1.00 |
| Integrated risk score without ML points | 209 | 1.00 | 1.00 | 1.00 | 63% | 77% | 93% | 100% | 54 | 0.98 | 1.00 |

## Price amendments recorded 5 days after first invoice

| Ordering | Queue | P@25 | P@50 | P@100 | Exposure @25 | @50 | @100 | @200 | Cases to 80% of exposure | Queue precision | Line recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rules, by severity label (previous default) | 300 | 1.00 | 1.00 | 0.96 | 63% | 83% | 96% | 99% | 44 | 0.68 | 1.00 |
| Rules, by exposure | 300 | 1.00 | 1.00 | 0.95 | 67% | 84% | 96% | 100% | 43 | 0.68 | 1.00 |
| ML only (Isolation Forest score) | 2020 | 1.00 | 1.00 | 0.81 | 41% | 47% | 55% | 63% | 355 | 0.11 | 1.00 |
| Integrated risk score (rules + ML) | 305 | 1.00 | 0.96 | 0.95 | 54% | 76% | 90% | 100% | 67 | 0.67 | 1.00 |
| Integrated risk score without ML points | 305 | 1.00 | 1.00 | 0.95 | 63% | 77% | 91% | 100% | 54 | 0.67 | 1.00 |

## Legitimate 35% renegotiation (3 suppliers)

| Ordering | Queue | P@25 | P@50 | P@100 | Exposure @25 | @50 | @100 | @200 | Cases to 80% of exposure | Queue precision | Line recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rules, by severity label (previous default) | 249 | 1.00 | 1.00 | 1.00 | 63% | 83% | 96% | 100% | 45 | 0.82 | 0.99 |
| Rules, by exposure | 249 | 1.00 | 1.00 | 1.00 | 67% | 84% | 96% | 100% | 43 | 0.82 | 0.99 |
| ML only (Isolation Forest score) | 2020 | 1.00 | 1.00 | 0.79 | 42% | 49% | 55% | 62% | 365 | 0.11 | 1.00 |
| Integrated risk score (rules + ML) | 258 | 1.00 | 1.00 | 1.00 | 54% | 77% | 91% | 100% | 67 | 0.79 | 0.99 |
| Integrated risk score without ML points | 258 | 1.00 | 1.00 | 1.00 | 63% | 77% | 93% | 100% | 54 | 0.79 | 0.99 |

## Returns posted as negative receipts (5% of receipts)

| Ordering | Queue | P@25 | P@50 | P@100 | Exposure @25 | @50 | @100 | @200 | Cases to 80% of exposure | Queue precision | Line recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rules, by severity label (previous default) | 297 | 1.00 | 0.92 | 0.74 | 63% | 80% | 91% | 98% | 49 | 0.69 | 1.00 |
| Rules, by exposure | 297 | 1.00 | 0.92 | 0.73 | 67% | 81% | 91% | 98% | 49 | 0.69 | 1.00 |
| ML only (Isolation Forest score) | 2020 | 1.00 | 1.00 | 0.80 | 41% | 46% | 55% | 68% | 332 | 0.11 | 1.00 |
| Integrated risk score (rules + ML) | 304 | 1.00 | 0.98 | 0.91 | 53% | 74% | 88% | 95% | 75 | 0.67 | 1.00 |
| Integrated risk score without ML points | 304 | 0.96 | 0.90 | 0.66 | 62% | 74% | 83% | 95% | 79 | 0.67 | 1.00 |

## ALL SEVEN COMBINED

| Ordering | Queue | P@25 | P@50 | P@100 | Exposure @25 | @50 | @100 | @200 | Cases to 80% of exposure | Queue precision | Line recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rules, by severity label (previous default) | 437 | 1.00 | 0.88 | 0.70 | 63% | 79% | 89% | 97% | 55 | 0.46 | 0.97 |
| Rules, by exposure | 437 | 0.88 | 0.86 | 0.70 | 63% | 79% | 89% | 97% | 54 | 0.46 | 0.97 |
| ML only (Isolation Forest score) | 2020 | 1.00 | 0.92 | 0.73 | 43% | 47% | 54% | 61% | 359 | 0.11 | 1.00 |
| Integrated risk score (rules + ML) | 456 | 0.96 | 0.88 | 0.81 | 53% | 69% | 84% | 95% | 73 | 0.45 | 0.98 |
| Integrated risk score without ML points | 456 | 0.88 | 0.82 | 0.61 | 60% | 74% | 82% | 94% | 90 | 0.45 | 0.98 |
