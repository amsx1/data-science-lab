"""DATA AUTOPSY analysis engine.

The package is deliberately free of any UI dependency so that every analytical
step can be unit-tested and reused from the dashboard, the CLI and notebooks:

* :mod:`src.data_loader` — robust CSV loading and column classification
* :mod:`src.quality` — missingness, duplicates, low variance, dtype drift
* :mod:`src.statistics` — descriptive statistics for numeric columns
* :mod:`src.categorical` — cardinality, dominance and entropy
* :mod:`src.outliers` — IQR, z-score and modified z-score rules
* :mod:`src.correlations` — Pearson/Spearman/Kendall + Cramer's V
* :mod:`src.anomalies` — Isolation Forest screening
* :mod:`src.leakage` — heuristic leakage *signals* (never verdicts)
* :mod:`src.health` — transparent, weighted dataset health score
* :mod:`src.cleaning` — opt-in, previewable transformations that export a separate CSV
* :mod:`src.investigation` — the orchestrator that runs all of the above
* :mod:`src.reporting` — Markdown and HTML report writers
* :mod:`src.charts` — Plotly figure builders (no Streamlit import)
* :mod:`src.ui_theme` — dashboard CSS
* :mod:`src.ui_sections` — Streamlit rendering functions
"""

__version__ = "1.0.0"

