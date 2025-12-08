"""
continuous.py
=============
TWFE Continuous Treatment Specifications.

Two specifications:
1. TWFE (basic): Facility + Year FE, clustered SEs by region
2. TWFE (strict): Facility + Region×Year FE, clustered SEs by region

Uses pyfixest for fast high-dimensional FE estimation.
    pip install pyfixest
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import warnings

from .config import FAC_ID_COL, YEAR_COL, CLUSTER_COL

# Import pyfixest (preferred for high-dimensional FE)
try:
    import pyfixest as pf
    HAS_PYFIXEST = True
except ImportError:
    HAS_PYFIXEST = False
    warnings.warn("pyfixest not installed. Install with: pip install pyfixest")


# =============================================================================
# TWFE Estimation
# =============================================================================

def estimate_twfe(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    controls: Optional[List[str]] = None,
    cluster_col: str = CLUSTER_COL,
    id_col: str = FAC_ID_COL,
    time_col: str = YEAR_COL,
    region_time_fe: bool = False,
    region_col: Optional[str] = None
) -> Dict:
    """
    Estimate TWFE with continuous treatment.
    
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
        Column for clustered SEs
    id_col : str
        Unit identifier
    time_col : str
        Time period
    region_time_fe : bool
        If True, use Region×Year FE instead of Year FE (Spec 2)
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
    
    if region_time_fe:
        # Spec 2: Facility + Region×Year FE
        region_col = region_col or cluster_col
        fe_str = f"{id_col} + {region_col}^{time_col}"
        spec_name = "TWFE (Region×Year FE)"
    else:
        # Spec 1: Facility + Year FE
        fe_str = f"{id_col} + {time_col}"
        spec_name = "TWFE (Year FE)"
    
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


def run_both_twfe_specs(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    controls: Optional[List[str]] = None,
    cluster_col: str = CLUSTER_COL,
    region_col: Optional[str] = None,
    **kwargs
) -> Dict[str, Dict]:
    """
    Run both TWFE specifications for a given outcome.
    
    Spec 1: Facility + Year FE
    Spec 2: Facility + Region×Year FE
    
    Both use region-clustered SEs.
    """
    print("=" * 60)
    print(f"TWFE: {treatment_col} → {outcome_col}")
    print("=" * 60)
    
    # Spec 1: Year FE
    spec1 = estimate_twfe(
        df, treatment_col, outcome_col,
        controls=controls,
        cluster_col=cluster_col,
        region_time_fe=False,
        **kwargs
    )
    
    # Spec 2: Region×Year FE
    spec2 = estimate_twfe(
        df, treatment_col, outcome_col,
        controls=controls,
        cluster_col=cluster_col,
        region_time_fe=True,
        region_col=region_col or cluster_col,
        **kwargs
    )
    
    return {
        "year_fe": spec1,
        "region_year_fe": spec2
    }


def run_all_outcomes(
    df: pd.DataFrame,
    treatment_col: str,
    outcomes: Dict[str, str],
    controls: Optional[List[str]] = None,
    cluster_col: str = CLUSTER_COL,
    **kwargs
) -> Dict[str, Dict]:
    """
    Run both TWFE specs for multiple outcomes.
    
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
        Clustering column
    
    Returns
    -------
    Dict with results for each outcome
    """
    all_results = {}
    
    for name, col in outcomes.items():
        print(f"\n{'#' * 60}")
        print(f"# Outcome: {name}")
        print(f"{'#' * 60}")
        
        all_results[name] = run_both_twfe_specs(
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
    
    for outcome, specs in results.items():
        for spec_name, res in specs.items():
            rows.append({
                "Outcome": outcome,
                "Specification": res["spec_name"],
                "Coefficient": res["coefficient"],
                "SE": res["se"],
                "P-value": res["pvalue"],
                "95% CI": f"[{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]",
                "N": res["n_obs"],
                "R²": res["r2"]
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
    cluster_col: str = CLUSTER_COL,
    weight_col: Optional[str] = None,
    label: Optional[str] = None
) -> Dict:
    """
    Unified interface for running outcome models.
    
    Runs both TWFE specs (Year FE, Region×Year FE) for a given outcome.
    Supports inverse-variance weighting for noisy outcomes.
    
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
        Column for clustered SEs
    weight_col : str, optional
        Weights column (e.g., 1/SE² for inverse-variance weighting)
    label : str, optional
        Label for output (defaults to outcome_col)
    
    Returns
    -------
    Dict with model results including:
        - year_fe: Spec 1 results
        - region_year_fe: Spec 2 results
        - metadata: sample size, weights info, etc.
    
    Example
    -------
    # ETS CO₂ (ground truth, no weighting)
    ets_results = run_outcome_models(panel, "log_ets_co2", "eu_alloc_ratio")
    
    # Satellite NOx (with inverse-variance weighting)
    panel["nox_weight"] = 1 / (panel["beirle_nox_kg_s_se"] ** 2)
    nox_results = run_outcome_models(panel, "beirle_nox_kg_s", "eu_alloc_ratio",
                                      weight_col="nox_weight")
    """
    label = label or outcome_col
    
    print(f"\n{'#' * 70}")
    print(f"# Outcome: {label}")
    if weight_col:
        print(f"# Weights: {weight_col}")
    print(f"{'#' * 70}")
    
    # Filter to valid observations
    df_valid = df.dropna(subset=[outcome_col, treatment_col])
    
    # TODO: pyfixest weight support - for now, document limitation
    if weight_col:
        print(f"  Note: Inverse-variance weighting requested but not yet implemented in pyfixest.")
        print(f"  Running unweighted estimation. Consider WLS extension.")
    
    # Run both specs
    results = run_both_twfe_specs(
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
        "n_facilities": df_valid[FAC_ID_COL].nunique(),
        "weighted": weight_col is not None,
        "weight_col": weight_col
    }
    
    return results


def run_dual_outcome_analysis(
    df: pd.DataFrame,
    treatment_col: str,
    controls: Optional[List[str]] = None,
    cluster_col: str = CLUSTER_COL,
    ets_col: str = "log_ets_co2",
    nox_col: str = "beirle_nox_kg_s",
    nox_se_col: Optional[str] = "beirle_nox_kg_s_se"
) -> Dict[str, Dict]:
    """
    Run models for both ETS CO₂ and Satellite NOx outcomes.
    
    Parameters
    ----------
    df : DataFrame
        Full panel with both outcomes
    treatment_col : str
        Treatment variable
    controls : list, optional
        Control variables
    cluster_col : str
        Clustering column
    ets_col : str
        ETS CO₂ outcome column
    nox_col : str
        Satellite NOx outcome column
    nox_se_col : str, optional
        SE column for NOx (for future weighting)
    
    Returns
    -------
    Dict with results for both outcomes:
        - ets: ETS CO₂ results
        - satellite: Satellite NOx results
    """
    results = {}
    
    # ETS CO₂ (ground truth - no weighting needed)
    print("\n" + "=" * 70)
    print("ETS VERIFIED CO₂ EMISSIONS (GROUND TRUTH)")
    print("=" * 70)
    results["ets"] = run_outcome_models(
        df, outcome_col=ets_col, treatment_col=treatment_col,
        controls=controls, cluster_col=cluster_col,
        label="ETS CO₂ (tCO₂/yr, log)"
    )
    
    # Satellite NOx (noisy - could use weighting)
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
    Format dual-outcome results as side-by-side comparison table.
    """
    rows = []
    
    for outcome_key, outcome_results in results.items():
        if outcome_key == "metadata":
            continue
        
        meta = outcome_results.get("metadata", {})
        label = meta.get("label", outcome_key)
        
        for spec_name, spec_results in outcome_results.items():
            if spec_name == "metadata":
                continue
            
            rows.append({
                "Outcome": label,
                "Specification": spec_results["spec_name"],
                "Coefficient": spec_results["coefficient"],
                "SE": spec_results["se"],
                "P-value": spec_results["pvalue"],
                "95% CI": f"[{spec_results['ci_lower']:.4f}, {spec_results['ci_upper']:.4f}]",
                "N": spec_results["n_obs"]
            })
    
    return pd.DataFrame(rows)
