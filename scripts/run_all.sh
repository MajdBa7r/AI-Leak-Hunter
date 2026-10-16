#!/usr/bin/env bash
# Reproduce every output and report from the procurement CSVs (about 8 minutes on a laptop).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
python -m leak_hunter.pipeline --out-dir outputs                                   # detection: findings, cases, scores
python -m leak_hunter.evaluation --out-dir outputs --report reports/detection_evaluation.md
python scripts/rules_noise_benchmark.py --out reports_rules/noise_benchmark.md     # rules under 7 disruptions
python scripts/unsupervised_benchmark.py --out reports_ml                          # 5 unsupervised detectors
python -m leak_hunter.queue_evaluation --out reports_queue                          # rules vs ML vs integrated queues
python scripts/make_replica.py --seed 2026                                          # untouched validation replica
python -m leak_hunter.queue_evaluation --data-dir data_replica_seed2026 --out reports_queue/replica_seed2026
echo "Done. Start the app with:  streamlit run app/streamlit_app.py"
