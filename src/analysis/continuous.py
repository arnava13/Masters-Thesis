"""
continuous.py
=============
TWFE Continuous Treatment Specification.

Single specification: Facility + Region×Year FE, clustered SEs by NUTS2 region.

Rationale: Region×Year FE absorbs regional time-varying confounders (electricity demand,
fuel prices, policy enforcement) that affect both treatment assignment and outcomes.

Uses pyfixest for fast high-dimensional FE estimation.
    pip install pyfixest
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import pyfixest as pf

# Default column names (can be overridden via function parameters)
_DEFAULT_FAC_ID_COL = "idx"
_DEFAULT_YEAR_COL = "year"
_DEFAULT_CLUSTER_COL = "nuts2_region"



# =============================================================================
# TWFE Estimation
# =============================================================================

def estimate_twfe(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    controls: Optional[List[str]] = None,
    cluster_col: str = _DEFAULT_CLUSTER_COL,
    id_col: str = _DEFAULT_FAC_ID_COL,
    time_col: str = _DEFAULT_YEAR_COL,
    region_col: Optional[str] = None
) -> Dict:
    """
    Estimate TWFE with continuous treatment.
    
    Uses Facility + Region×Year FE with clustered SEs by region.
    
    Parameters
    ----------
    df : DataFrame
        Panel data
    treatment_col : str
        Continuous treatment variable (e.g., eu_alloc_ratio)
    outcome_col : str
        Outcome variable (e.g., log_ets_co2 or beirle_nox_kg_s)
    controls : list, optional
        Control variables
    cluster_col : str
        Column for clustered SEs (NUTS2 region)
    id_col : str
        Unit identifier
    time_col : str
        Time period
    region_col : str, optional
        Region column for Region×Year FE (defaults to cluster_col)
    
    Returns
    -------
    Dict with coefficient, SE, p-value, CI, and model details
    """
    if not HAS_PYFIXEST:
        raise ImportError("pyfixest required. Install: pip install pyfixest")
    
    df = df.copy()
    df = df.dropna(subset=[outcome_col, treatment_col])
    
    # Build formula
    controls = controls or []
    control_str = " + ".join(controls) if controls else ""
    
    # Facility + Region×Year FE
    region_col = region_col or cluster_col
    fe_str = f"{id_col} + {region_col}^{time_col}"
    spec_name = "TWFE (Facility + Region×Year FE)"
    
    # Full formula: outcome ~ treatment + controls | FE
    if control_str:
        formula = f"{outcome_col} ~ {treatment_col} + {control_str} | {fe_str}"
    else:
        formula = f"{outcome_col} ~ {treatment_col} | {fe_str}"
    
    print(f"\n{spec_name}")
    print(f"  Formula: {formula}")
    print(f"  Cluster: {cluster_col}")
    
    # Estimate with pyfixest
    model = pf.feols(formula, data=df, vcov={"CRV1": cluster_col})
    
    # Extract results
    coef = model.coef().get(treatment_col, np.nan)
    se = model.se().get(treatment_col, np.nan)
    pval = model.pvalue().get(treatment_col, np.nan)
    
    results = {
        "spec_name": spec_name,
        "coefficient": coef,
        "se": se,
        "pvalue": pval,
        "ci_lower": coef - 1.96 * se,
        "ci_upper": coef + 1.96 * se,
        "n_obs": model.nobs,
        "r2": model.r2,
        "formula": formula,
        "cluster_col": cluster_col,
        "treatment_col": treatment_col,
        "outcome_col": outcome_col,
        "model": model
    }
    
    print(f"  Coef: {coef:.6f} (SE: {se:.6f}, p={pval:.4f})")
    
    return results


def run_twfe(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    controls: Optional[List[str]] = None,
    cluster_col: str = _DEFAULT_CLUSTER_COL,
    region_col: Optional[str] = None,
    **kwargs
) -> Dict:
    """
    Run TWFE specification for a given outcome.
    
    Uses Facility + Region×Year FE with clustered SEs.
    """
    print("=" * 60)
    print(f"TWFE: {treatment_col} -> {outcome_col}")
    print("=" * 60)
    
    return estimate_twfe(
        df, treatment_col, outcome_col,
        controls=controls,
        cluster_col=cluster_col,
        region_col=region_col or cluster_col,
        **kwargs
    )


def run_all_outcomes(
    df: pd.DataFrame,
    treatment_col: str,
    outcomes: Dict[str, str],
    controls: Optional[List[str]] = None,
    cluster_col: str = _DEFAULT_CLUSTER_COL,
    **kwargs
) -> Dict[str, Dict]:
    """
    Run TWFE for multiple outcomes.
    
    Parameters
    ----------
    df : DataFrame
        Panel data
    treatment_col : str
        Treatment variable
    outcomes : dict
        {name: column} mapping, e.g., {"NOx": "beirle_nox_kg_s"}
    controls : list
        Control variables
    cluster_col : str
        Clustering column (NUTS2 region)
    
    Returns
    -------
    Dict with results for each outcome
    """
    all_results = {}
    
    for name, col in outcomes.items():
        print(f"\n{'#' * 60}")
        print(f"# Outcome: {name}")
        print(f"{'#' * 60}")
        
        all_results[name] = run_twfe(
            df, treatment_col, col,
            controls=controls,
            cluster_col=cluster_col,
            **kwargs
        )
    
    return all_results


# =============================================================================
# Results Formatting
# =============================================================================

def format_results_table(results: Dict) -> pd.DataFrame:
    """
    Format estimation results as a table.
    """
    rows = []
    
    for outcome, res in results.items():
        if isinstance(res, dict) and "coefficient" in res:
            rows.append({
                "Outcome": outcome,
                "Specification": res.get("spec_name", "TWFE"),
                "Coefficient": res["coefficient"],
                "SE": res["se"],
                "P-value": res["pvalue"],
                "95% CI": f"[{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]",
                "N": res["n_obs"],
                "R2": res.get("r2", np.nan)
            })
    
    return pd.DataFrame(rows)


def summarize_results(results: Dict, title: str = "Results"):
    """
    Print formatted summary of results.
    """
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")
    
    df = format_results_table(results)
    print(df.to_string(index=False))


# =============================================================================
# Unified Outcome Model Interface
# =============================================================================

def run_outcome_models(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    controls: Optional[List[str]] = None,
    cluster_col: str = _DEFAULT_CLUSTER_COL,
    weight_col: Optional[str] = None,
    label: Optional[str] = None
) -> Dict:
    """
    Unified interface for running outcome models.
    
    Runs TWFE (Facility + Region×Year FE) for a given outcome.
    
    Parameters
    ----------
    df : DataFrame
        Panel data with treatment and outcome
    outcome_col : str
        Outcome variable column name
    treatment_col : str
        Treatment variable (continuous)
    controls : list, optional
        Control variables
    cluster_col : str
        Column for clustered SEs (NUTS2 region)
    weight_col : str, optional
        Weights column (not yet implemented)
    label : str, optional
        Label for output (defaults to outcome_col)
    
    Returns
    -------
    Dict with model results and metadata
    """
    label = label or outcome_col
    
    print(f"\n{'#' * 70}")
    print(f"# Outcome: {label}")
    if weight_col:
        print(f"# Weights: {weight_col}")
    print(f"{'#' * 70}")
    
    # Filter to valid observations
    df_valid = df.dropna(subset=[outcome_col, treatment_col])
    
    if weight_col:
        print(f"  Note: Inverse-variance weighting not yet implemented.")
    
    # Run TWFE
    results = run_twfe(
        df_valid,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        controls=controls,
        cluster_col=cluster_col
    )
    
    # Add metadata
    results["metadata"] = {
        "label": label,
        "outcome_col": outcome_col,
        "treatment_col": treatment_col,
        "n_obs": len(df_valid),
        "n_facilities": df_valid[_DEFAULT_FAC_ID_COL].nunique(),
        "weighted": weight_col is not None,
        "weight_col": weight_col
    }
    
    return results


def run_dual_outcome_analysis(
    df: pd.DataFrame,
    treatment_col: str,
    controls: Optional[List[str]] = None,
    cluster_col: str = _DEFAULT_CLUSTER_COL,
    ets_col: str = "log_ets_co2",
    nox_col: str = "beirle_nox_kg_s",
    nox_se_col: Optional[str] = "beirle_nox_kg_s_se"
) -> Dict[str, Dict]:
    """
    Run TWFE for both ETS CO2 and Satellite NOx outcomes.
    
    Uses single specification: Facility + Region×Year FE with clustered SEs.
    """
    results = {}
    
    # ETS CO2 (ground truth)
    print("\n" + "=" * 70)
    print("ETS VERIFIED CO2 EMISSIONS (GROUND TRUTH)")
    print("=" * 70)
    results["ets"] = run_outcome_models(
        df, outcome_col=ets_col, treatment_col=treatment_col,
        controls=controls, cluster_col=cluster_col,
        label="ETS CO2 (tCO2/yr, log)"
    )
    
    # Satellite NOx
    print("\n" + "=" * 70)
    print("SATELLITE NOx EMISSION PROXY (BEIRLE-STYLE)")
    print("=" * 70)
    results["satellite"] = run_outcome_models(
        df, outcome_col=nox_col, treatment_col=treatment_col,
        controls=controls, cluster_col=cluster_col,
        label="Satellite NOx (kg/s)"
    )
    
    return results


def format_dual_results_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Format dual-outcome results as comparison table.
    """
    rows = []
    
    for outcome_key, outcome_results in results.items():
        if outcome_key == "metadata":
            continue
        
        meta = outcome_results.get("metadata", {})
        label = meta.get("label", outcome_key)
        
        # Single spec per outcome now
        if "coefficient" in outcome_results:
            rows.append({
                "Outcome": label,
                "Coefficient": outcome_results["coefficient"],
                "SE": outcome_results["se"],
                "P-value": outcome_results["pvalue"],
                "95% CI": f"[{outcome_results['ci_lower']:.4f}, {outcome_results['ci_upper']:.4f}]",
                "N": outcome_results["n_obs"]
            })
    
    return pd.DataFrame(rows)
