"""
csdid.py
========
Callaway-Sant'Anna Difference-in-Differences Estimator.

Reference:
    Callaway, B., & Sant'Anna, P. H. (2021). "Difference-in-Differences with 
    Multiple Time Periods." Journal of Econometrics, 225(2), 200-230.
    https://doi.org/10.1016/j.jeconom.2020.12.001

Uses the canonical `csdid` Python package (port of the R `did` package):
    pip install csdid

Treatment definition:
    - Treated = eu_alloc_ratio < 1 (facility must purchase allowances)
    - Cohort = first year when alloc_ratio < 1
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
import warnings
import matplotlib.pyplot as plt

from .config import (
    FAC_ID_COL, YEAR_COL, CLUSTER_COL,
    CS_ANTICIPATION, CS_CONTROL_GROUP, CS_EST_METHOD
)

# Import csdid
try:
    import csdid
    HAS_CSDID = True
except ImportError:
    HAS_CSDID = False
    warnings.warn(
        "csdid not installed. Install with: pip install csdid\n"
        "Reference: https://github.com/d2cml-ai/csdid"
    )


# =============================================================================
# Cohort Construction
# =============================================================================

def build_cohorts(
    df: pd.DataFrame,
    treatment_col: str = "treated",
    cohort_col: str = "cohort",
    id_col: str = FAC_ID_COL,
    time_col: str = YEAR_COL
) -> pd.DataFrame:
    """
    Build cohort assignments for Callaway-Sant'Anna.
    
    Cohort = first year a unit is treated. Units never treated get cohort=0.
    
    Parameters
    ----------
    df : DataFrame
        Panel data with binary treatment indicator
    treatment_col : str
        Binary treatment column (1 = treated in that year)
    cohort_col : str
        Output column name for cohort
    
    Returns
    -------
    DataFrame with cohort column added
    """
    df = df.copy()
    
    # First treatment year per unit
    first_treated = (
        df[df[treatment_col] == 1]
        .groupby(id_col)[time_col]
        .min()
        .rename(cohort_col)
    )
    
    # Merge back
    if cohort_col in df.columns:
        df = df.drop(columns=[cohort_col])
    
    df = df.merge(first_treated, on=id_col, how="left")
    df[cohort_col] = df[cohort_col].fillna(0).astype(int)
    
    # Summary
    cohort_counts = df.groupby(id_col)[cohort_col].first().value_counts().sort_index()
    print("Cohort distribution:")
    for cohort, count in cohort_counts.items():
        label = "Never treated" if cohort == 0 else f"Cohort {cohort}"
        print(f"  {label}: {count} facilities")
    
    return df


def validate_cohorts(
    df: pd.DataFrame,
    cohort_col: str = "cohort",
    id_col: str = FAC_ID_COL
) -> Dict:
    """
    Validate cohort data for CS-DiD estimation.
    
    Returns dict with validation results and warnings.
    """
    results = {"valid": True, "warnings": [], "cohort_summary": {}}
    
    cohort_fac = df.groupby(id_col)[cohort_col].first()
    cohort_counts = cohort_fac.value_counts().sort_index()
    
    n_never_treated = cohort_counts.get(0, 0)
    n_ever_treated = cohort_counts[cohort_counts.index > 0].sum()
    n_cohorts = (cohort_counts.index > 0).sum()
    
    results["cohort_summary"] = {
        "n_never_treated": int(n_never_treated),
        "n_ever_treated": int(n_ever_treated),
        "n_cohorts": int(n_cohorts),
        "cohort_sizes": cohort_counts.to_dict()
    }
    
    if n_never_treated < 10:
        results["warnings"].append(
            f"Only {n_never_treated} never-treated. Consider 'notyettreated' control."
        )
    
    if n_cohorts < 2:
        results["warnings"].append(f"Only {n_cohorts} cohort(s). Limited identification.")
        results["valid"] = False
    
    for w in results["warnings"]:
        print(f"Warning: {w}")
    
    if results["valid"]:
        print(f"Valid: {n_cohorts} cohorts, {n_ever_treated} treated, {n_never_treated} control")
    
    return results


# =============================================================================
# Callaway-Sant'Anna Estimation
# =============================================================================

def estimate_callaway_santanna(
    df: pd.DataFrame,
    outcome_col: str,
    cohort_col: str = "cohort",
    id_col: str = FAC_ID_COL,
    time_col: str = YEAR_COL,
    xformla: Optional[str] = None,
    control_group: str = CS_CONTROL_GROUP,
    anticipation: int = CS_ANTICIPATION,
    est_method: str = CS_EST_METHOD,
    cluster_col: Optional[str] = None
) -> Dict:
    """
    Estimate Callaway-Sant'Anna group-time ATT.
    
    Parameters
    ----------
    df : DataFrame
        Panel data
    outcome_col : str
        Outcome variable name
    cohort_col : str
        Cohort column (first treatment year, 0 = never treated)
    id_col : str
        Unit identifier
    time_col : str
        Time period
    xformla : str, optional
        R-style formula for covariates (e.g., "~ x1 + x2")
    control_group : str
        "nevertreated" or "notyettreated"
    anticipation : int
        Anticipation periods (usually 0)
    est_method : str
        "dr" (doubly robust), "ipw", or "reg"
    cluster_col : str, optional
        Column for clustered SEs (e.g., pypsa_cluster)
    
    Returns
    -------
    Dict with att_gt, aggregations, and metadata
    """
    if not HAS_CSDID:
        raise ImportError("csdid not installed. Run: pip install csdid")
    
    df = df.copy()
    
    # Ensure proper types
    df[id_col] = df[id_col].astype(int)
    df[time_col] = df[time_col].astype(int)
    df[cohort_col] = df[cohort_col].astype(int)
    
    # Drop missing outcomes
    df = df.dropna(subset=[outcome_col])
    
    print(f"\nEstimating CS-DiD:")
    print(f"  Outcome: {outcome_col}")
    print(f"  Control group: {control_group}")
    print(f"  Estimation method: {est_method}")
    print(f"  Clustered SEs: {cluster_col if cluster_col else 'None (unit-level)'}")
    
    # Estimate ATT(g,t)
    # Note: csdid clustervars syntax may vary by version
    att_gt_result = csdid.att_gt(
        data=df,
        yname=outcome_col,
        gname=cohort_col,
        tname=time_col,
        idname=id_col,
        xformla=xformla,
        control_group=control_group,
        anticipation=anticipation,
        est_method=est_method
    )
    
    # Aggregate: simple (overall ATT)
    agg_simple = csdid.aggte(att_gt_result, type="simple")
    
    # Aggregate: dynamic (event study)
    agg_dynamic = csdid.aggte(att_gt_result, type="dynamic")
    
    # Aggregate: by cohort
    agg_group = csdid.aggte(att_gt_result, type="group")
    
    # Extract results
    results = {
        "att_gt": att_gt_result,
        "agg_simple": {
            "att": agg_simple.overall_att,
            "se": agg_simple.overall_se,
            "ci": (
                agg_simple.overall_att - 1.96 * agg_simple.overall_se,
                agg_simple.overall_att + 1.96 * agg_simple.overall_se
            )
        },
        "agg_dynamic": agg_dynamic,
        "agg_group": agg_group,
        "n_obs": len(df),
        "n_units": df[id_col].nunique(),
        "n_cohorts": df[df[cohort_col] > 0][cohort_col].nunique(),
        "outcome": outcome_col,
        "cluster_col": cluster_col
    }
    
    print(f"\nResults:")
    print(f"  Overall ATT: {results['agg_simple']['att']:.6f}")
    print(f"  SE: {results['agg_simple']['se']:.6f}")
    print(f"  95% CI: [{results['agg_simple']['ci'][0]:.6f}, {results['agg_simple']['ci'][1]:.6f}]")
    
    return results


def run_csdid_both_specs(
    df: pd.DataFrame,
    outcome_col: str,
    cohort_col: str = "cohort",
    cluster_col: str = CLUSTER_COL,
    **kwargs
) -> Dict[str, Dict]:
    """
    Run CS-DiD with both clustering specifications:
    1. Without region clustering (unit-level SEs)
    2. With region-clustered SEs
    
    Parameters
    ----------
    df : DataFrame
        Panel data
    outcome_col : str
        Outcome variable
    cohort_col : str
        Cohort column
    cluster_col : str
        Column for regional clustering
    **kwargs
        Additional arguments passed to estimate_callaway_santanna
    
    Returns
    -------
    Dict with 'no_cluster' and 'region_cluster' results
    """
    print("=" * 60)
    print(f"CS-DiD: {outcome_col}")
    print("=" * 60)
    
    # Spec 1: No clustering (unit-level)
    print("\n--- Specification 1: Unit-level SEs ---")
    results_no_cluster = estimate_callaway_santanna(
        df, outcome_col, cohort_col,
        cluster_col=None,
        **kwargs
    )
    
    # Spec 2: Region-clustered
    print(f"\n--- Specification 2: {cluster_col}-clustered SEs ---")
    results_clustered = estimate_callaway_santanna(
        df, outcome_col, cohort_col,
        cluster_col=cluster_col,
        **kwargs
    )
    
    return {
        "no_cluster": results_no_cluster,
        "region_cluster": results_clustered
    }


# =============================================================================
# Event Study Plots
# =============================================================================

def plot_event_study(
    results: Dict,
    title: Optional[str] = None,
    xlabel: str = "Years Relative to Treatment",
    ylabel: str = "ATT",
    figsize: Tuple[int, int] = (10, 6),
    max_pre: int = 3,
    max_post: int = 4
) -> plt.Figure:
    """
    Plot event study from CS-DiD results.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract dynamic results
    dyn = results["agg_dynamic"]
    
    if hasattr(dyn, "egt"):
        event_times = np.array(dyn.egt)
        atts = np.array(dyn.att_egt)
        ses = np.array(dyn.se_egt)
    else:
        raise ValueError("Cannot extract event study data")
    
    # Filter to desired range
    mask = (event_times >= -max_pre) & (event_times <= max_post)
    event_times = event_times[mask]
    atts = atts[mask]
    ses = ses[mask]
    
    # Plot with error bars
    ax.errorbar(
        event_times, atts,
        yerr=1.96 * ses,
        fmt="o-",
        capsize=3,
        markersize=6,
        color="steelblue",
        ecolor="steelblue",
        alpha=0.8
    )
    
    # Reference lines
    ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax.axvline(-0.5, color="red", linestyle="--", linewidth=1, alpha=0.7)
    
    # Shade pre-treatment
    ax.axvspan(-max_pre - 0.5, -0.5, alpha=0.1, color="gray")
    
    # Labels
    outcome = results.get("outcome", "Outcome")
    if title is None:
        title = f"Event Study: {outcome}"
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(-max_pre, max_post + 1))
    
    plt.tight_layout()
    return fig


def test_pre_trends(results: Dict, n_pre_periods: int = 2) -> Dict:
    """
    Test for parallel pre-trends (joint null that pre-treatment ATTs = 0).
    """
    from scipy import stats
    
    dyn = results["agg_dynamic"]
    
    if hasattr(dyn, "egt"):
        event_times = np.array(dyn.egt)
        atts = np.array(dyn.att_egt)
        ses = np.array(dyn.se_egt)
    else:
        return {"error": "Cannot extract event study data"}
    
    # Pre-treatment periods
    pre_mask = (event_times < 0) & (event_times >= -n_pre_periods)
    pre_atts = atts[pre_mask]
    pre_ses = ses[pre_mask]
    
    if len(pre_atts) == 0:
        return {"error": "No pre-treatment periods"}
    
    # t-stats and Wald test
    t_stats = pre_atts / pre_ses
    wald_stat = np.sum(t_stats ** 2)
    df = len(pre_atts)
    p_value = 1 - stats.chi2.cdf(wald_stat, df)
    
    result = {
        "wald_statistic": wald_stat,
        "df": df,
        "p_value": p_value,
        "reject_null_05": p_value < 0.05
    }
    
    if result["reject_null_05"]:
        print(f"Pre-trends: REJECT null at 5% (p={p_value:.4f}) - caution advised")
    else:
        print(f"Pre-trends: Fail to reject null (p={p_value:.4f}) - parallel trends OK")
    
    return result
