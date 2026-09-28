"""Central configuration for DATA AUTOPSY.

Every numeric threshold, weight and label used during an investigation lives here
so that the behaviour of the tool is auditable and tunable in exactly one place.
Nothing in this module is dataset-specific: the same thresholds are applied to
every uploaded file, which keeps the analysis reproducible and honest.

Thresholds are expressed as *fractions* (0.05 == 5%) unless the name says
otherwise.
"""

from __future__ import annotations

from typing import Final

# --------------------------------------------------------------------------- #
# Missing data
# --------------------------------------------------------------------------- #
#: Below this share of missing values a column is considered healthy.
MISSING_LOW: Final[float] = 0.05
#: Above this share the column deserves active investigation.
MISSING_MODERATE: Final[float] = 0.20
#: Above this share imputation becomes statistically dubious.
MISSING_HIGH: Final[float] = 0.50
#: Above this share the column carries almost no information.
MISSING_CRITICAL: Final[float] = 0.80

# --------------------------------------------------------------------------- #
# Duplicate records
# --------------------------------------------------------------------------- #
DUPLICATE_LOW: Final[float] = 0.001
DUPLICATE_MODERATE: Final[float] = 0.01
DUPLICATE_HIGH: Final[float] = 0.05

# --------------------------------------------------------------------------- #
# Constant / low-variance features
# --------------------------------------------------------------------------- #
#: A column whose most frequent value covers this share of rows is near-constant.
NEAR_CONSTANT_DOMINANCE: Final[float] = 0.99
#: Two distinct values is the minimum for a column to have any discriminative power.
MIN_DISTINCT_VALUES: Final[int] = 2

# --------------------------------------------------------------------------- #
# Outliers
# --------------------------------------------------------------------------- #
#: Classic Tukey fence multiplier applied to the interquartile range.
IQR_MULTIPLIER: Final[float] = 1.5
#: Extreme Tukey fence (far outliers).
IQR_EXTREME_MULTIPLIER: Final[float] = 3.0
#: |z| beyond which an observation is flagged by the z-score method.
ZSCORE_THRESHOLD: Final[float] = 3.0
#: |z| threshold for the robust (median absolute deviation) z-score.
MODIFIED_ZSCORE_THRESHOLD: Final[float] = 3.5
#: Smallest sample for which z-scores are statistically meaningful.
MIN_ROWS_FOR_ZSCORE: Final[int] = 8
OUTLIER_RATE_LOW: Final[float] = 0.01
OUTLIER_RATE_MODERATE: Final[float] = 0.05
OUTLIER_RATE_HIGH: Final[float] = 0.15
#: Minimum distinct values required before a column enters outlier testing. Binary
#: indicators and near-binary flags are excluded: the IQR/z-score rules would simply
#: re-detect the minority class and report it as an "outlier".
MIN_UNIQUE_FOR_OUTLIER_TESTS: Final[int] = 4
#: |skewness| above which a distribution is reported as clearly asymmetric.
SKEW_THRESHOLD: Final[float] = 1.0
#: Excess kurtosis above which tails are reported as heavy.
KURTOSIS_THRESHOLD: Final[float] = 3.0

# --------------------------------------------------------------------------- #
# Correlation
# --------------------------------------------------------------------------- #
CORR_STRONG: Final[float] = 0.70
CORR_MODERATE: Final[float] = 0.50
CORR_NEAR_PERFECT: Final[float] = 0.95
#: Pairs backed by fewer complete observations than this are marked unreliable.
MIN_PAIRWISE_OBSERVATIONS: Final[int] = 10
#: Cap on the number of pairs for which a p-value is computed (keeps the UI fast).
MAX_PAIRS_FOR_PVALUE: Final[int] = 5_000
#: |r| above which two features are treated as redundant for linear models.
MULTICOLLINEARITY_FLAG: Final[float] = 0.80
#: Cramer's V above which a categorical association is reported as strong.
CRAMERS_V_STRONG: Final[float] = 0.50
#: Cramer's V above which a categorical association is reported as very strong.
CRAMERS_V_VERY_STRONG: Final[float] = 0.70
#: Maximum number of categorical columns entering the association screen.
MAX_ASSOCIATION_COLUMNS: Final[int] = 15
#: Maximum number of levels a categorical column may have to enter the screen.
MAX_ASSOCIATION_LEVELS: Final[int] = 40

# --------------------------------------------------------------------------- #
# Categorical variables
# --------------------------------------------------------------------------- #
#: A single category above this share is reported as dominant.
DOMINANT_CATEGORY: Final[float] = 0.90
#: A single category above this share is reported as severely dominant.
DOMINANT_CATEGORY_SEVERE: Final[float] = 0.98
#: Categories below this share are reported as rare.
RARE_CATEGORY_SHARE: Final[float] = 0.01
#: Absolute unique-value count that makes a categorical column high-cardinality.
HIGH_CARDINALITY_UNIQUE: Final[int] = 50
#: Distinct values per row above which a column looks like free text.
HIGH_CARDINALITY_RATIO: Final[float] = 0.50
#: Above this share of unique values a column is treated as an identifier.
IDENTIFIER_UNIQUE_RATIO: Final[float] = 0.95

# --------------------------------------------------------------------------- #
# Data-type / consistency checks
# --------------------------------------------------------------------------- #
#: Share of values that must parse as numbers before a text column is flagged.
NUMERIC_STRING_RATIO: Final[float] = 0.90
#: Share of values that must parse as dates before a text column is flagged.
DATETIME_STRING_RATIO: Final[float] = 0.90
#: Rows sampled when probing a text column for hidden numbers/dates.
TYPE_PROBE_SAMPLE: Final[int] = 1_000

# --------------------------------------------------------------------------- #
# Potential data leakage heuristics
# --------------------------------------------------------------------------- #
#: Absolute correlation with the target above which leakage is plausible.
TARGET_CORRELATION_FLAG: Final[float] = 0.98
#: Unique-value share above which a feature looks like a record identifier.
LEAKAGE_ID_UNIQUE_RATIO: Final[float] = 0.95
#: Whole-word column-name tokens that typically denote identifiers. Matching is
#: token-based (see ``src.utils.tokenize_name``), so "gender" never matches "end".
ID_NAME_TOKENS: Final[tuple[str, ...]] = (
    "id", "uuid", "guid", "index", "row", "key", "identifier", "record", "pid",
    "serial", "ref", "reference", "number", "no", "num", "code", "customer",
    "patient", "student", "user", "account", "email", "phone", "mobile", "ssn",
    "passport", "national", "address", "name", "hash", "token", "roll", "reg",
)
#: Whole-word tokens that commonly denote outcome information (moderate confidence).
OUTCOME_NAME_TOKENS: Final[tuple[str, ...]] = (
    "target", "label", "outcome", "result", "final", "after", "post", "resolved",
    "closed", "approved", "decision", "converted", "churned", "repaid", "default",
    "survived", "diagnosis", "response", "winner", "outcome_flag",
)
#: Whole-word tokens that mark measurements of the outcome itself (weak signal:
#: reported at informational level only).
WEAK_OUTCOME_NAME_TOKENS: Final[tuple[str, ...]] = (
    "score", "grade", "status", "class", "rank", "total", "amount", "value",
)
#: Whole-word tokens that unambiguously mark a post-outcome *temporal* value.
POST_OUTCOME_TOKENS: Final[tuple[str, ...]] = (
    "after", "post", "final", "end", "exit", "close", "closed", "resolved",
    "followup", "closure", "next",
)
#: Tokens that only indicate post-outcome information when the column is a status or a
#: date (a count such as "assignments_completed" is an ordinary feature, not leakage).
POST_OUTCOME_CONDITIONAL_TOKENS: Final[tuple[str, ...]] = (
    "completion", "completed", "graduated", "survival", "survived", "settled",
)
#: Accuracy at which a single feature perfectly separates a binary target.
PERFECT_SEPARATION_ACCURACY: Final[float] = 1.0
#: Correlation above which two features are treated as duplicates of each other.
DUPLICATE_FEATURE_CORRELATION: Final[float] = 0.995

# --------------------------------------------------------------------------- #
# Anomaly detection (Isolation Forest)
# --------------------------------------------------------------------------- #
ANOMALY_MIN_ROWS: Final[int] = 20
ANOMALY_MIN_FEATURES: Final[int] = 2
ANOMALY_MAX_ROWS: Final[int] = 50_000
ANOMALY_DEFAULT_CONTAMINATION: Final[float] = 0.05
ANOMALY_CONTAMINATION_MIN: Final[float] = 0.001
ANOMALY_CONTAMINATION_MAX: Final[float] = 0.50
ANOMALY_DEFAULT_ESTIMATORS: Final[int] = 200
RANDOM_STATE: Final[int] = 42

# --------------------------------------------------------------------------- #
# Health score
# --------------------------------------------------------------------------- #
#: Weights of the five measurable health dimensions. They sum to 1.0.
HEALTH_WEIGHTS: Final[dict[str, float]] = {
    "completeness": 0.30,
    "integrity": 0.20,
    "consistency": 0.20,
    "uniqueness": 0.15,
    "variability": 0.15,
}
#: Outlier cell-rate at which the integrity sub-score is fully deducted.
INTEGRITY_SATURATION_RATE: Final[float] = 0.05
#: Missing cell-rate at which the completeness sub-score is fully deducted.
COMPLETENESS_SATURATION_RATE: Final[float] = 0.25
#: Duplicate row-rate at which the uniqueness sub-score is fully deducted.
UNIQUENESS_SATURATION_RATE: Final[float] = 0.10
#: (minimum score, grade label) bands, evaluated from best to worst.
HEALTH_GRADES: Final[tuple[tuple[int, str], ...]] = (
    (90, "Excellent"),
    (75, "Good"),
    (60, "Fair"),
    (40, "Poor"),
    (0, "Critical"),
)

# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #
SEVERITY_COLORS: Final[dict[str, str]] = {
    "info": "#58A6FF",
    "low": "#3FB950",
    "moderate": "#D29922",
    "high": "#F0883E",
    "critical": "#F85149",
}
SEVERITY_ICONS: Final[dict[str, str]] = {
    "info": "i",
    "low": "!",
    "moderate": "!!",
    "high": "!!!",
    "critical": "!!!!",
}
#: Severity order used for sorting findings (most severe first).
SEVERITY_RANK: Final[dict[str, int]] = {
    "critical": 4,
    "high": 3,
    "moderate": 2,
    "low": 1,
    "info": 0,
}

#: Maximum number of example records shown for a single finding.
MAX_EXAMPLE_ROWS: Final[int] = 10
#: Maximum number of columns listed by name inside a sentence before truncating.
MAX_COLUMNS_IN_TEXT: Final[int] = 8
