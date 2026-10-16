# Investigation-queue evaluation

Unit = case (an incident; for ML-only, a single invoice line). Window = invoice lines after the Isolation Forest training cutoff (out-of-time). Labels are used only to score the queues. Risk-score weights were fixed in `config.PrioritizationConfig` before this evaluation was first run and have not been changed since. Single run per scenario (seeded); no confidence intervals.

## Clean data (no disruption)

| Ordering | Queue | P@25 | P@50 | P@100 | Exposure @25 | @50 | @100 | @200 | Cases to 80% of exposure | Queue precision | Line recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rules, by severity label (previous default) | 213 | 1.00 | 1.00 | 1.00 | 59% | 83% | 97% | 100% | 45 | 1.00 | 1.00 |
| Rules, by exposure | 213 | 1.00 | 1.00 | 1.00 | 67% | 84% | 97% | 100% | 43 | 1.00 | 1.00 |
| ML only (Isolation Forest score) | 2004 | 1.00 | 1.00 | 0.84 | 32% | 53% | 60% | 78% | 227 | 0.12 | 1.00 |
| Integrated risk score (rules + ML) | 228 | 1.00 | 1.00 | 1.00 | 61% | 75% | 94% | 100% | 61 | 0.93 | 1.00 |
| Integrated risk score without ML points | 228 | 1.00 | 1.00 | 1.00 | 61% | 83% | 95% | 100% | 45 | 0.93 | 1.00 |
| Expected value (exposure x confidence), added after v1 results | 228 | 1.00 | 1.00 | 1.00 | 67% | 84% | 97% | 100% | 43 | 0.93 | 1.00 |

## Fuzzy duplicate invoice numbers (all 40 re-entries)

| Ordering | Queue | P@25 | P@50 | P@100 | Exposure @25 | @50 | @100 | @200 | Cases to 80% of exposure | Queue precision | Line recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rules, by severity label (previous default) | 213 | 1.00 | 1.00 | 1.00 | 59% | 83% | 97% | 100% | 45 | 1.00 | 1.00 |
| Rules, by exposure | 213 | 1.00 | 1.00 | 1.00 | 67% | 84% | 97% | 100% | 43 | 1.00 | 1.00 |
| ML only (Isolation Forest score) | 2004 | 1.00 | 1.00 | 0.84 | 32% | 53% | 60% | 78% | 227 | 0.12 | 1.00 |
| Integrated risk score (rules + ML) | 228 | 1.00 | 1.00 | 1.00 | 61% | 75% | 94% | 100% | 60 | 0.93 | 1.00 |
| Integrated risk score without ML points | 228 | 1.00 | 1.00 | 1.00 | 61% | 83% | 95% | 100% | 45 | 0.93 | 1.00 |
| Expected value (exposure x confidence), added after v1 results | 228 | 1.00 | 1.00 | 1.00 | 67% | 84% | 97% | 100% | 43 | 0.93 | 1.00 |

## 25% of re-entries booked under a duplicate vendor record

| Ordering | Queue | P@25 | P@50 | P@100 | Exposure @25 | @50 | @100 | @200 | Cases to 80% of exposure | Queue precision | Line recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rules, by severity label (previous default) | 213 | 1.00 | 1.00 | 1.00 | 62% | 83% | 97% | 100% | 45 | 1.00 | 0.98 |
| Rules, by exposure | 213 | 1.00 | 1.00 | 1.00 | 67% | 84% | 97% | 100% | 43 | 1.00 | 0.98 |
| ML only (Isolation Forest score) | 2004 | 1.00 | 1.00 | 0.85 | 33% | 40% | 62% | 78% | 210 | 0.12 | 1.00 |
| Integrated risk score (rules + ML) | 226 | 1.00 | 1.00 | 1.00 | 63% | 75% | 94% | 100% | 60 | 0.94 | 0.98 |
| Integrated risk score without ML points | 226 | 1.00 | 1.00 | 1.00 | 63% | 83% | 95% | 100% | 45 | 0.94 | 0.98 |
| Expected value (exposure x confidence), added after v1 results | 226 | 1.00 | 1.00 | 1.00 | 67% | 84% | 97% | 100% | 43 | 0.94 | 0.98 |

## Warehouse posting lag 21 days

| Ordering | Queue | P@25 | P@50 | P@100 | Exposure @25 | @50 | @100 | @200 | Cases to 80% of exposure | Queue precision | Line recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rules, by severity label (previous default) | 215 | 1.00 | 1.00 | 0.98 | 59% | 83% | 97% | 100% | 45 | 0.99 | 1.00 |
| Rules, by exposure | 215 | 1.00 | 0.98 | 0.98 | 67% | 84% | 97% | 100% | 43 | 0.99 | 1.00 |
| ML only (Isolation Forest score) | 2004 | 1.00 | 0.98 | 0.80 | 32% | 53% | 60% | 74% | 236 | 0.12 | 1.00 |
| Integrated risk score (rules + ML) | 239 | 1.00 | 1.00 | 0.98 | 61% | 75% | 93% | 100% | 62 | 0.89 | 1.00 |
| Integrated risk score without ML points | 239 | 1.00 | 0.96 | 0.98 | 61% | 82% | 94% | 100% | 45 | 0.89 | 1.00 |
| Expected value (exposure x confidence), added after v1 results | 239 | 1.00 | 0.98 | 0.98 | 67% | 84% | 97% | 100% | 43 | 0.89 | 1.00 |

## Rounding noise (+/-1 cent, 3 decimals) on 10% of lines

| Ordering | Queue | P@25 | P@50 | P@100 | Exposure @25 | @50 | @100 | @200 | Cases to 80% of exposure | Queue precision | Line recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rules, by severity label (previous default) | 213 | 1.00 | 1.00 | 1.00 | 59% | 83% | 97% | 100% | 45 | 1.00 | 1.00 |
| Rules, by exposure | 213 | 1.00 | 1.00 | 1.00 | 67% | 84% | 97% | 100% | 43 | 1.00 | 1.00 |
| ML only (Isolation Forest score) | 2004 | 1.00 | 1.00 | 0.86 | 34% | 53% | 61% | 73% | 243 | 0.12 | 1.00 |
| Integrated risk score (rules + ML) | 227 | 1.00 | 1.00 | 1.00 | 62% | 75% | 93% | 100% | 61 | 0.94 | 1.00 |
| Integrated risk score without ML points | 227 | 1.00 | 1.00 | 1.00 | 61% | 83% | 95% | 100% | 45 | 0.94 | 1.00 |
| Expected value (exposure x confidence), added after v1 results | 227 | 1.00 | 1.00 | 1.00 | 67% | 84% | 97% | 100% | 43 | 0.94 | 1.00 |

## Price amendments recorded 5 days after first invoice

| Ordering | Queue | P@25 | P@50 | P@100 | Exposure @25 | @50 | @100 | @200 | Cases to 80% of exposure | Queue precision | Line recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rules, by severity label (previous default) | 303 | 1.00 | 1.00 | 0.96 | 59% | 83% | 96% | 100% | 45 | 0.70 | 1.00 |
| Rules, by exposure | 303 | 1.00 | 0.96 | 0.96 | 67% | 83% | 96% | 100% | 43 | 0.70 | 1.00 |
| ML only (Isolation Forest score) | 2004 | 1.00 | 1.00 | 0.84 | 34% | 52% | 61% | 77% | 245 | 0.12 | 1.00 |
| Integrated risk score (rules + ML) | 317 | 1.00 | 1.00 | 1.00 | 61% | 74% | 94% | 100% | 61 | 0.67 | 1.00 |
| Integrated risk score without ML points | 317 | 1.00 | 1.00 | 1.00 | 61% | 83% | 94% | 100% | 45 | 0.67 | 1.00 |
| Expected value (exposure x confidence), added after v1 results | 317 | 1.00 | 1.00 | 0.96 | 67% | 84% | 96% | 100% | 43 | 0.67 | 1.00 |

## Legitimate 35% renegotiation (3 suppliers)

| Ordering | Queue | P@25 | P@50 | P@100 | Exposure @25 | @50 | @100 | @200 | Cases to 80% of exposure | Queue precision | Line recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rules, by severity label (previous default) | 220 | 1.00 | 1.00 | 1.00 | 59% | 83% | 97% | 100% | 45 | 0.95 | 0.99 |
| Rules, by exposure | 220 | 1.00 | 1.00 | 1.00 | 67% | 84% | 97% | 100% | 43 | 0.95 | 0.99 |
| ML only (Isolation Forest score) | 2004 | 1.00 | 1.00 | 0.83 | 32% | 53% | 60% | 77% | 227 | 0.12 | 1.00 |
| Integrated risk score (rules + ML) | 235 | 1.00 | 1.00 | 1.00 | 61% | 75% | 94% | 100% | 60 | 0.89 | 0.99 |
| Integrated risk score without ML points | 235 | 1.00 | 1.00 | 1.00 | 61% | 83% | 95% | 100% | 45 | 0.89 | 0.99 |
| Expected value (exposure x confidence), added after v1 results | 235 | 1.00 | 1.00 | 1.00 | 67% | 84% | 97% | 100% | 43 | 0.89 | 0.99 |

## Returns posted as negative receipts (5% of receipts)

| Ordering | Queue | P@25 | P@50 | P@100 | Exposure @25 | @50 | @100 | @200 | Cases to 80% of exposure | Queue precision | Line recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rules, by severity label (previous default) | 303 | 0.96 | 0.94 | 0.80 | 57% | 81% | 93% | 98% | 49 | 0.70 | 1.00 |
| Rules, by exposure | 303 | 0.92 | 0.94 | 0.80 | 64% | 82% | 93% | 99% | 47 | 0.70 | 1.00 |
| ML only (Isolation Forest score) | 2004 | 1.00 | 1.00 | 0.83 | 32% | 53% | 60% | 74% | 240 | 0.12 | 1.00 |
| Integrated risk score (rules + ML) | 318 | 1.00 | 1.00 | 0.95 | 61% | 74% | 91% | 98% | 64 | 0.67 | 1.00 |
| Integrated risk score without ML points | 318 | 0.96 | 0.96 | 0.78 | 59% | 81% | 90% | 98% | 48 | 0.67 | 1.00 |
| Expected value (exposure x confidence), added after v1 results | 318 | 0.92 | 0.94 | 0.80 | 64% | 82% | 93% | 99% | 47 | 0.67 | 1.00 |

## ALL SEVEN COMBINED

| Ordering | Queue | P@25 | P@50 | P@100 | Exposure @25 | @50 | @100 | @200 | Cases to 80% of exposure | Queue precision | Line recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Rules, by severity label (previous default) | 388 | 1.00 | 0.94 | 0.77 | 62% | 81% | 92% | 98% | 48 | 0.54 | 0.97 |
| Rules, by exposure | 388 | 1.00 | 0.94 | 0.77 | 67% | 82% | 93% | 98% | 47 | 0.54 | 0.97 |
| ML only (Isolation Forest score) | 2004 | 1.00 | 0.96 | 0.77 | 32% | 39% | 60% | 72% | 229 | 0.12 | 1.00 |
| Integrated risk score (rules + ML) | 410 | 1.00 | 0.98 | 0.92 | 61% | 74% | 90% | 98% | 66 | 0.51 | 0.97 |
| Integrated risk score without ML points | 410 | 1.00 | 0.94 | 0.79 | 63% | 81% | 91% | 97% | 49 | 0.51 | 0.97 |
| Expected value (exposure x confidence), added after v1 results | 410 | 1.00 | 0.94 | 0.79 | 67% | 82% | 93% | 98% | 47 | 0.51 | 0.97 |
