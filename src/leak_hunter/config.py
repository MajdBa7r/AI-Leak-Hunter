"""Configuration for the AI Leak Hunter detection engine.

Every threshold, tolerance and model parameter used by the rules and the
Isolation Forest lives here, so a run is fully described by one
``DetectionConfig`` (it is written to ``run_summary.json`` with every run).

None of these defaults were tuned against the ground-truth labels:

* The historical-spike settings (5 prior observations, 1.30 x median) mirror
  the rule definition documented with the dataset.
* Materiality (business policy, not label-tuned): price flags need a line
  overcharge > $5 AND variance > 2%; quantity flags need > 1.5% and >= $25;
  posting-lag grace period 30 days.
* ``review_budget`` is a reviewer-capacity assumption (share of invoice lines
  an analyst can triage), not an estimate of how many anomalies exist.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Union


@dataclass(frozen=True)
class RuleConfig:
    # Rule A - duplicate invoices
    invoice_number_strip_leading_zeros: bool = False
    """Strip leading zeros (per separator-delimited token) in the EXACT key too.
    The fuzzy key always ignores leading zeros."""
    dup_fuzzy: bool = True
    """Also link documents whose numbers differ by look-alike characters,
    leading zeros or one edit, when amount and date also match."""
    dup_fuzzy_max_distance: int = 2
    """Max optimal-string-alignment distance (insert/delete/substitute/adjacent
    swap) between fuzzy keys."""
    dup_fuzzy_min_length: int = 5
    """Edit-distance matching only for keys at least this long (short numbers
    collide too easily); equal fuzzy keys always match."""
    dup_fuzzy_amount_tolerance: float = 0.0
    """USD difference allowed between document totals for a fuzzy match."""
    dup_fuzzy_window_days: int = 60
    """Max days between invoice dates for a fuzzy match."""
    dup_fuzzy_allow_digit_substitution: bool = False
    """Allow numbers that differ only by digit-for-digit substitutions (INV-1041
    vs INV-1052) as fuzzy matches. Off by default: sequentially numbered
    recurring invoices look exactly like that. Adjacent digit swaps are allowed."""
    dup_fuzzy_require_same_items: bool = True
    """Fuzzy matches also need the same set of line-item descriptions."""
    dup_recurring_interval_days: int = 30
    """Fuzzy (not exact-number) matches exactly this many days apart are treated
    as recurring billing (subscriptions, retainers) and not linked. 0 disables."""

    # Rule B - quantity discrepancies
    qty_tolerance_units: int = 0
    """Units of excess tolerated before a line is flagged."""
    qty_materiality_pct: float = 0.015
    """Ignore quantity excesses below this share of the cumulative invoiced
    quantity on the PO line (1.5%)."""
    qty_materiality_value: float = 25.0
    """Ignore quantity excesses worth less than this (excess units x approved
    price, USD) - low-value lines are not worth an investigation."""
    qty_grace_days: int = 30
    """Posting-lag grace period. While an invoice is younger than this, units
    billed beyond the receipts posted so far (but within the approved quantity)
    are reported as PENDING, not flagged. Units beyond the approved quantity are
    flagged immediately, because no later receipt can justify them."""
    as_of_date: str | None = None
    """Extraction date used for invoice ages ("YYYY-MM-DD"). Default: the latest
    invoice or receipt date in the data."""

    # Rule C - PO price mismatches (also used for Rule D's contractual excess)
    price_tolerance_abs: float = 0.01
    """USD per unit tolerated above the approved price (rounding allowance)."""
    price_tolerance_pct: float = 0.005
    """Relative tolerance above the approved price (rounding); used by Rule D's
    contractual-excess test."""
    price_materiality_abs: float = 5.00
    """Rule C materiality (dual threshold): flag only when the LINE overcharge
    (unit difference x quantity) exceeds this USD amount ..."""
    price_materiality_pct: float = 0.02
    """... AND the unit-price variance exceeds this share of the approved price
    (2%, contractually allowed rounding / FX fluctuation). The exposure is still
    the full difference x quantity."""
    amendment_borderline_business_days: int = 3
    """An approved price amendment dated within this many business days AFTER the
    invoice is treated as a timing ambiguity: the finding is kept but marked
    confidence LOW / BORDERLINE_AMENDMENT_TIMING."""
    pending_amendment_statuses: tuple = ("pending", "draft", "submitted", "in review", "in_review")
    """Non-final amendment statuses: never authorise a price, but a line billed at
    such a requested price gets confidence MEDIUM instead of HIGH."""

    # Rule D - historical supplier price spikes
    hist_min_prior: int = 5
    """Minimum number of valid, strictly earlier observations for a baseline."""
    hist_baseline: str = "median"
    """Baseline statistic over prior unit prices: "median" (robust) or "mean"."""
    hist_ratio_threshold: float = 1.30
    """Flag when unit_price > threshold x baseline."""
    hist_adaptive: bool = True
    """Raise the threshold for suppliers whose prices are naturally variable:
    effective ratio = max(hist_ratio_threshold, 1 + k x robust CV), where
    robust CV = 1.4826 x MAD / median of the prior prices."""
    hist_dispersion_k: float = 4.0
    hist_exclude_duplicate_reentries: bool = True
    """Do not count re-entered (duplicate-number) invoices as independent
    historical purchases."""


@dataclass(frozen=True)
class IsolationForestConfig:
    enabled: bool = True
    n_estimators: int = 300
    max_samples: Union[int, float, str] = "auto"
    max_features: float = 1.0
    contamination: Union[float, str] = "auto"
    """Passed to scikit-learn. Only affects the model's own predict() offset,
    which this engine reports but does not use for review decisions. The
    synthetic dataset's 10% planted-anomaly share is deliberately NOT used."""
    random_state: int = 42
    fit_mode: str = "temporal"
    """"temporal": fit on the earliest ``train_fraction`` of invoice lines (by
    invoice date) and score everything; "all": fit on all lines (no labels are
    used either way, but temporal mode allows an out-of-time evaluation)."""
    train_fraction: float = 0.60
    review_budget: float = 0.02
    """Share of invoice lines analysts can review. The review threshold is the
    (1 - review_budget) quantile of TRAINING-window anomaly scores."""
    supplier_history_min_prior: int = 5
    frequency_window_days: int = 30
    explain_top_n: int = 3


@dataclass(frozen=True)
class SeverityConfig:
    high_exposure: float = 5000.0
    medium_exposure: float = 500.0
    if_high_percentile: float = 0.995


@dataclass(frozen=True)
class PrioritizationConfig:
    """Transparent, points-based case priority (max 100). Weights are a priori
    business judgement, fixed BEFORE the queue evaluation was run, and never
    fitted to labels. Every case shows its point breakdown."""
    exposure_points: float = 35.0
    """Scaled by log10(1 + exposure) / log10(1 + exposure_cap)."""
    exposure_cap: float = 50_000.0
    evidence_points: dict = field(default_factory=lambda: {"HIGH": 30.0, "MEDIUM": 20.0, "LOW": 8.0,
                                                           "UNUSUAL_ONLY": 8.0})
    """Rule violation with a contractual basis, by rule confidence; UNUSUAL_ONLY =
    statistical or historical anomaly with no contractual basis."""
    confirmed_loss_points: float = 15.0
    paid_exposure_points: float = 7.0
    """Money already out of the door makes recovery urgent."""
    anomaly_points: float = 20.0
    anomaly_floor_percentile: float = 0.90
    """Isolation Forest points rise linearly from this TRAINING-score percentile to 1.0."""
    confidence_value_factor: dict = field(default_factory=lambda: {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.25})
    """EXPECTED_VALUE ordering: exposure x factor (no contractual basis -> 0)."""
    queue_order: str = "expected_value"
    """Default queue order: "risk_score" (v1, pre-registered) or "expected_value".
    Chosen by the pre-committed rule in reports_queue/DECISION_RULE.md."""
    bands: tuple = ((60.0, "P1"), (35.0, "P2"), (0.0, "P3"))
    band_labels: dict = field(default_factory=lambda: {"P1": "Investigate now", "P2": "Review this cycle",
                                                       "P3": "Monitor"})


@dataclass(frozen=True)
class DetectionConfig:
    rules: RuleConfig = field(default_factory=RuleConfig)
    isolation_forest: IsolationForestConfig = field(default_factory=IsolationForestConfig)
    severity: SeverityConfig = field(default_factory=SeverityConfig)
    prioritization: PrioritizationConfig = field(default_factory=PrioritizationConfig)
    currency: str = "USD"

    def to_dict(self) -> dict:
        return asdict(self)
