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
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure # type: ignore
from csdid.att_gt import ATTgt

# Default values (can be overridden via function parameters)
FAC_ID_COL = "idx"
YEAR_COL = "year"
CLUSTER_COL = "nuts2_region"
CS_ANTICIPATION = 0
CS_CONTROL_GROUP = "notyettreated"
CS_EST_METHOD = "dr"

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
        .rename(cohort_col) # type: ignore
    ) # type: ignore
    
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
        "n_never_treated": int(n_never_treated), # type: ignore
        "n_ever_treated": int(n_ever_treated),
        "n_cohorts": int(n_cohorts),
        "cohort_sizes": cohort_counts.to_dict()
    }
    
    if n_never_treated < 10: # type: ignore
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
    cluster_col : str, optional
        Column for clustered SEs
    
    Returns
    -------
    Dict with att_gt, aggregations, and metadata
    """
    
    df = df.copy()
    
    # Ensure proper types
    df[id_col] = df[id_col].astype(int)
    df[time_col] = df[time_col].astype(int)
    df[cohort_col] = df[cohort_col].astype(int)
    
    # Drop missing outcomes
    df = df.dropna(subset=[outcome_col])
    
    # Filter out small cohorts (< 10 units) to avoid estimation issues
    cohort_sizes = df[df[cohort_col] > 0].groupby(cohort_col)[id_col].nunique()
    small_cohorts = cohort_sizes[cohort_sizes < 10].index.tolist() # type: ignore
    if small_cohorts:
        print(f"\n  Dropping small cohorts (<10 units): {small_cohorts}")
        # Set small cohort units to never-treated (cohort=0) so they join control pool
        df.loc[df[cohort_col].isin(small_cohorts), cohort_col] = 0
        remaining_cohorts = df[df[cohort_col] > 0][cohort_col].unique() # type: ignore
        print(f"  Remaining cohorts: {sorted(remaining_cohorts)}")
    
    print(f"\nEstimating CS-DiD:")
    print(f"  Outcome: {outcome_col}")
    print(f"  Control group: {control_group}")
    print(f"  Covariates: {xformla if xformla else 'None'}")
    print(f"  N obs: {len(df)}, N units: {df[id_col].nunique()}")
    
    # Check we have enough cohorts
    n_cohorts = df[df[cohort_col] > 0][cohort_col].nunique() # type: ignore
    if n_cohorts < 1:
        print("  ERROR: No valid cohorts remaining after filtering")
        return {"error": "No valid cohorts", "n_obs": len(df)}
    
    # Estimate ATT(g,t) with error handling
    try:
        att_gt_model = ATTgt(
            yname=outcome_col,
            gname=cohort_col,
            tname=time_col,
            idname=id_col,
            xformla=xformla, # type: ignore
            data=df,
            control_group=control_group,
            anticipation=anticipation,
            clustervar=cluster_col,
            allow_unbalanced_panel=True
        ).fit()
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"error": str(e), "n_obs": len(df), "n_units": df[id_col].nunique()}
    
    # Aggregate: simple (overall ATT)
    att_gt_model.aggte("simple")
    
    # Get simple aggregation results from the model's internal state
    # The aggte method updates the model in place
    simple_results = att_gt_model.aggte("simple")
    
    # Extract results
    results = {
        "att_gt": att_gt_model,
        "agg_simple": {
            "att": att_gt_model.att,  # Overall ATT after aggte("simple") # type: ignore
            "se": att_gt_model.se, # type: ignore
            "ci": (
                att_gt_model.att - 1.96 * att_gt_model.se if att_gt_model.att is not None else np.nan, # type: ignore
                att_gt_model.att + 1.96 * att_gt_model.se if att_gt_model.att is not None else np.nan # type: ignore
            )
        },
        "n_obs": len(df),
        "n_units": df[id_col].nunique(),
        "n_cohorts": df[df[cohort_col] > 0][cohort_col].nunique(), # type: ignore
        "outcome": outcome_col
    }
    
    att = results['agg_simple']['att']
    se = results['agg_simple']['se']
    if att is not None and se is not None:
        print(f"\nResults:")
        print(f"  Overall ATT: {att:.6f}")
        print(f"  SE: {se:.6f}")
        print(f"  95% CI: [{att - 1.96*se:.6f}, {att + 1.96*se:.6f}]")
    else:
        print(f"\nResults: Could not compute ATT")
    
    return results


def run_csdid_outcome(
    df: pd.DataFrame,
    outcome_col: str,
    cohort_col: str = "cohort",
    cluster_col: str = CLUSTER_COL,
    **kwargs
) -> Dict:
    """
    Run CS-DiD for a single outcome with region-clustered SEs.
    
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
    Dict with CS-DiD results
    """
    print("=" * 60)
    print(f"CS-DiD: {outcome_col}")
    print("=" * 60)
    
    return estimate_callaway_santanna(
        df, outcome_col, cohort_col,
        cluster_col=cluster_col,
        **kwargs
    )


def run_csdid_dual_outcome(
    df: pd.DataFrame,
    cohort_col: str = "cohort",
    cluster_col: str = CLUSTER_COL,
    ets_col: str = "log_ets_co2",
    nox_col: str = "beirle_nox_kg_s",
    nox_dl_col: str = "above_dl_0_03",
    embedding_n_components: int = 10,
    embedding_prefix: str = "emb_",
    **kwargs
) -> Dict[str, Dict]:
    """
    Run CS-DiD for both ETS CO2 and Satellite NOx outcomes.
    
    - ETS CO₂: Full sample, no embeddings
    - Satellite NOx: Detection-limit-filtered sample, with PCA and PLS embeddings
    
    IMPORTANT: NOx analysis ALWAYS applies detection limit filter.
    Samples below detection limit are excluded.
    
    Parameters
    ----------
    df : DataFrame
        Panel data with cohort assignments
    cohort_col : str
        Cohort column (first treatment year, 0 = never treated)
    cluster_col : str
        Column for regional clustering
    ets_col : str
        ETS CO2 outcome column (log-transformed)
    nox_col : str
        Satellite NOx outcome column
    nox_dl_col : str
        Detection limit column: 'above_dl_0_03' (permissive) or 'above_dl_0_11' (conservative)
    embedding_n_components : int
        Number of components for reduced embeddings (default: 10)
    embedding_prefix : str
        Prefix for raw embedding columns (default: 'emb_')
    **kwargs
        Additional arguments passed to estimate_callaway_santanna
    
    Returns
    -------
    Dict with keys: 'ets', 'satellite_pca', 'satellite_pls'
    """
    from embedding_reduction import reduce_embeddings, get_reduced_embedding_cols
    
    results = {}
    
    # ==========================================================================
    # ETS CO2 (ground truth) - FULL SAMPLE, NO EMBEDDINGS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("CS-DiD: ETS VERIFIED CO2 EMISSIONS (GROUND TRUTH)")
    print("  Sample: Full ETS panel")
    print("  Covariates: none (DR estimator uses outcome regression)")
    print("=" * 70)
    
    results["ets"] = estimate_callaway_santanna(
        df, ets_col, cohort_col,
        cluster_col=cluster_col,
        **kwargs
    )
    
    # ==========================================================================
    # Satellite NOx - DETECTION-LIMIT FILTERED, WITH EMBEDDINGS
    # NOx analysis NEVER uses unfiltered samples
    # ==========================================================================
    df_nox = df.copy()
    n_before = len(df_nox)
    
    # Apply detection limit filter (REQUIRED)
    if nox_dl_col in df_nox.columns:
        df_nox = df_nox[df_nox[nox_dl_col] == True]
        dl_label = "≥0.03 kg/s" if "0_03" in nox_dl_col else "≥0.11 kg/s"
    else:
        df_nox = df_nox.dropna(subset=[nox_col])
        dl_label = "non-missing"
    
    n_after = len(df_nox)
    print(f"\nNOx Detection Limit Filter ({dl_label}): {n_before} → {n_after} obs")
    
    if n_after < 50:
        print(f"WARNING: Insufficient NOx observations ({n_after}) after DL filter")
        return results
    
    raw_emb_cols = [c for c in df_nox.columns if c.startswith(embedding_prefix)]
    has_embeddings = len(raw_emb_cols) > 0
    
    for emb_method in ["pca", "pls"]:
        print("\n" + "=" * 70)
        print(f"CS-DiD: SATELLITE NOx ({emb_method.upper()}) — Detection Limit: {dl_label}")
        
        if has_embeddings:
            # Apply dimensionality reduction
            df_reduced = reduce_embeddings(
                df_nox.copy(), # type: ignore
                method=emb_method,
                n_components=embedding_n_components,
                target_col=nox_col if emb_method == "pls" else None,
                emb_prefix=embedding_prefix
            )
            emb_cols = get_reduced_embedding_cols(df_reduced, method=emb_method)
            
            # Build xformla for CS-DiD covariates
            xformla = "~ " + " + ".join(emb_cols)
            print(f"  Covariates: {emb_method.upper()} embeddings ({len(emb_cols)} dims)")
        else:
            df_reduced = df_nox
            xformla = None
            print("  Covariates: none (no embeddings available)")
        
        print("=" * 70)
        
        results[f"satellite_{emb_method}"] = estimate_callaway_santanna(
            df_reduced, nox_col, cohort_col, # type: ignore
            cluster_col=cluster_col,
            xformla=xformla,
            **kwargs
        )
    
    return results


def format_csdid_results_table(results: Dict[str, Dict]) -> "pd.DataFrame":
    """
    Format CS-DiD dual-outcome results as comparison table.
    """
    rows = []
    
    for outcome_key, outcome_results in results.items():
        if "agg_simple" not in outcome_results:
            continue
        
        agg = outcome_results["agg_simple"]
        label = outcome_key.replace("_", " ").title()
        
        rows.append({
            "Outcome": label,
            "ATT": agg["att"],
            "SE": agg["se"],
            "95% CI": f"[{agg['ci'][0]:.4f}, {agg['ci'][1]:.4f}]",
            "N": outcome_results.get("n_obs", "")
        })
    
    return pd.DataFrame(rows)


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
) -> Figure:
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
