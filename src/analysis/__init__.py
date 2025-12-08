"""
Analysis Module for Site-Level ETS Policy Effects Study
========================================================

Modular framework for causal inference on EU ETS policy effects.

Dual Outcomes:
- ETS CO₂: Verified annual emissions from EU ETS registry (ground-truth)
- Satellite NOx: Beirle-style flux-divergence proxy (noisy, with uncertainty)

Submodules:
-----------
- config: Paths and column name constants
- data: Load panel, build treatment variables, satellite filters
- clusters: PyPSA-Eur region assignment
- continuous: TWFE specifications (pyfixest)
- csdid: Callaway-Sant'Anna DiD (csdid package)
- diagnostics: Validation and visualization

Usage:
------
>>> from src.analysis.config import FAC_ID_COL, CLUSTER_COL, ETS_CO2_COL, NOX_OUTCOME_COL
>>> from src.analysis.data import load_analysis_panel, apply_satellite_filters
>>> from src.analysis.continuous import run_dual_outcome_analysis
"""

# Config exports
from .config import (
    FAC_ID_COL, YEAR_COL, CLUSTER_COL, ALLOC_RATIO_COL,
    # ETS outcome
    ETS_CO2_COL, LOG_ETS_CO2_COL,
    # Satellite NOx outcome
    NOX_OUTCOME_COL, NOX_SE_COL, NOX_REL_ERR_COL,
    DL_CONSERVATIVE_COL, DL_PERMISSIVE_COL,
    # Paths
    FACILITIES_STATIC_PATH, FACILITIES_YEARLY_PATH, BEIRLE_PANEL_PATH
)

# Data exports
from .data import (
    load_analysis_panel,
    load_facilities_static,
    load_beirle_panel,
    build_treatment_variables,
    apply_sample_filters,
    apply_satellite_filters,
    get_intersection_sample
)

# Clusters exports
from .clusters import (
    assign_pypsa_clusters,
    get_cluster_summary
)

# Continuous exports
from .continuous import (
    estimate_twfe,
    run_both_twfe_specs,
    run_all_outcomes,
    run_outcome_models,
    run_dual_outcome_analysis,
    format_results_table,
    format_dual_results_table
)

# CS-DiD exports
from .csdid import (
    build_cohorts,
    validate_cohorts,
    estimate_callaway_santanna,
    run_csdid_both_specs,
    plot_event_study,
    test_pre_trends
)

__all__ = [
    # Config
    "FAC_ID_COL", "YEAR_COL", "CLUSTER_COL", "ALLOC_RATIO_COL",
    "ETS_CO2_COL", "LOG_ETS_CO2_COL",
    "NOX_OUTCOME_COL", "NOX_SE_COL", "NOX_REL_ERR_COL",
    "DL_CONSERVATIVE_COL", "DL_PERMISSIVE_COL",
    # Data
    "load_analysis_panel", "load_facilities_static", "load_beirle_panel",
    "build_treatment_variables", "apply_sample_filters",
    "apply_satellite_filters", "get_intersection_sample",
    # Clusters
    "assign_pypsa_clusters", "get_cluster_summary",
    # Continuous
    "estimate_twfe", "run_both_twfe_specs", "run_all_outcomes",
    "run_outcome_models", "run_dual_outcome_analysis",
    "format_results_table", "format_dual_results_table",
    # CS-DiD
    "build_cohorts", "validate_cohorts", "estimate_callaway_santanna",
    "run_csdid_both_specs", "plot_event_study", "test_pre_trends",
]
