"""
continuous.py
=============
TWFE Continuous Treatment Specification.

Single specification: Facility + Region×Year FE, clustered SEs by NUTS2 region.

Rationale: Region×Year FE absorbs regional time-varying confounders (electricity demand,
fuel prices, policy enforcement) that affect both treatment assignment and outcomes.

Uses pyfixest for fast high-dimensional FE estimation.
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
    region_col: Optional[str] = None,
    weight_col: Optional[str] = None
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
    weight_col : str, optional
        Column containing observation weights (e.g., inverse-variance weights)
    
    Returns
    -------
    Dict with coefficient, SE, p-value, CI, and model details
    """
    
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
    if weight_col:
        print(f"  Weights: {weight_col}")
    
    # Estimate with pyfixest (with optional weights)
    if weight_col and weight_col in df.columns:
        model = pf.feols(formula, data=df, vcov={"CRV1": cluster_col}, weights=weight_col)  # type: ignore
    else:
        model = pf.feols(formula, data=df, vcov={"CRV1": cluster_col})
    
    # Extract results (with null safety)
    coef_dict = model.coef()
    se_dict = model.se()
    pval_dict = model.pvalue()
    
    coef = coef_dict.get(treatment_col, np.nan) if coef_dict is not None else np.nan
    se = se_dict.get(treatment_col, np.nan) if se_dict is not None else np.nan
    pval = pval_dict.get(treatment_col, np.nan) if pval_dict is not None else np.nan
    
    # Get performance metrics (nobs, r2) - handle different pyfixest versions
    n_obs = np.nan
    r2 = np.nan
    
    # Try get_performance() method (newer pyfixest)
    if hasattr(model, "get_performance"):
        perf = model.get_performance() # type: ignore
        if perf is not None and isinstance(perf, dict):
            n_obs = perf.get("nobs", np.nan)
            r2 = perf.get("r2", np.nan)
    
    # Fallback to internal attributes if still nan
    if np.isnan(n_obs) if isinstance(n_obs, float) else n_obs is None:
        for attr in ["_N", "nobs", "_nobs"]:
            if hasattr(model, attr):
                val = getattr(model, attr)
                if val is not None:
                    n_obs = val
                    break
    if np.isnan(r2) if isinstance(r2, float) else r2 is None:
        for attr in ["_r2", "r2", "_r2_within"]:
            if hasattr(model, attr):
                val = getattr(model, attr)
                if val is not None:
                    r2 = val
                    break
    
    
    results = {
        "spec_name": spec_name,
        "coefficient": coef,
        "se": se,
        "pvalue": pval,
        "ci_lower": coef - 1.96 * se, # type: ignore
        "ci_upper": coef + 1.96 * se, # type: ignore
        "n_obs": n_obs,
        "r2": r2,
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
        Column containing observation weights (e.g., inverse-variance weights for satellite NOx)
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
    
    # Also filter by weights if using weighted estimation
    if weight_col and weight_col in df_valid.columns:
        df_valid = df_valid.dropna(subset=[weight_col])
    
    # Run TWFE (with optional weights)
    results = run_twfe(
        df_valid,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        controls=controls,
        cluster_col=cluster_col,
        weight_col=weight_col if weight_col and weight_col in df_valid.columns else None
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


def run_ets_analysis(
    df: pd.DataFrame,
    treatment_col: str,
    controls: Optional[List[str]] = None,
    cluster_col: str = _DEFAULT_CLUSTER_COL,
    ets_col: str = "log_ets_co2",
) -> Dict:
    """
    Run TWFE for ETS CO2 outcome on full sample.
    
    ETS CO₂ uses NO embeddings because geographic context does not affect
    administrative data measurement. Confounders absorbed by FE.
    
    Returns
    -------
    Dict with TWFE results for ETS CO₂
    """
    print("\n" + "=" * 70)
    print("ETS VERIFIED CO2 EMISSIONS (GROUND TRUTH)")
    print("  Sample: Full ETS panel")
    print("  Controls: base only (embeddings NOT used - absorbed by FE)")
    print("=" * 70)
    
    return run_outcome_models(
        df, outcome_col=ets_col, treatment_col=treatment_col,
        controls=controls, cluster_col=cluster_col,
        label="ETS CO2 (tCO2/yr, log)"
    )


def run_nox_analysis(
    df: pd.DataFrame,
    treatment_col: str,
    controls: Optional[List[str]] = None,
    cluster_col: str = _DEFAULT_CLUSTER_COL,
    nox_col: str = "beirle_nox_kg_s",
    nox_dl_col: str = "above_dl_0_03",
    nox_weight_col: Optional[str] = "nox_weight",
    embedding_n_components: int = 10,
    embedding_prefix: str = "emb_"
) -> Dict[str, Dict]:
    """
    Run TWFE for Satellite NOx outcome with detection limit filter.
    
    IMPORTANT: NOx analysis ALWAYS requires detection limit filtering.
    Samples below detection limit (0.03 kg/s minimum) are excluded.
    
    Runs BOTH embedding reduction strategies (PCA and PLS) as robustness checks.
    
    Parameters
    ----------
    df : DataFrame
        Panel data (will be filtered by detection limit)
    nox_dl_col : str
        Detection limit column: 'above_dl_0_03' (permissive) or 'above_dl_0_11' (conservative)
    nox_weight_col : str, optional
        Column for inverse-variance weights
    embedding_n_components : int
        Number of PCA/PLS components (default: 10)
    
    Returns
    -------
    Dict with keys: 'satellite_pca', 'satellite_pls'
    """
    from embedding_reduction import reduce_embeddings, get_reduced_embedding_cols
    
    results = {}
    
    # ==========================================================================
    # FILTER BY DETECTION LIMIT (REQUIRED)
    # NOx analysis NEVER uses unfiltered samples
    # ==========================================================================
    df_nox = df.copy()
    n_before = len(df_nox)
    
    # Apply detection limit filter
    if nox_dl_col in df_nox.columns:
        df_nox = df_nox[df_nox[nox_dl_col] == True]
        dl_label = "≥0.03 kg/s" if "0_03" in nox_dl_col else "≥0.11 kg/s"
    else:
        # Fallback: require non-missing NOx
        df_nox = df_nox.dropna(subset=[nox_col])
        dl_label = "non-missing"
    
    n_after = len(df_nox)
    print(f"\nNOx Detection Limit Filter ({dl_label}): {n_before} → {n_after} obs")
    
    if n_after < 50:
        print(f"WARNING: Insufficient NOx observations ({n_after}) after DL filter")
        return results
    
    # Check if weights are available
    use_weights = nox_weight_col and nox_weight_col in df_nox.columns
    
    # Check for embeddings
    raw_emb_cols = [c for c in df_nox.columns if c.startswith(embedding_prefix)]
    has_embeddings = len(raw_emb_cols) > 0
    
    for emb_method in ["pca", "pls"]:
        nox_controls = list(controls) if controls else []
        
        emb_method_desc = ""
        if has_embeddings:
            # Check how many NOx obs have valid embeddings
            nox_valid = df_nox[nox_col].notna() # type: ignore
            emb_valid = df_nox[raw_emb_cols[0]].notna() # type: ignore
            n_both = (nox_valid & emb_valid).sum()
            
            if n_both > 0:
                # Apply dimensionality reduction (fresh copy each time)
                df_reduced = reduce_embeddings(
                    df_nox.copy(), # type: ignore
                    method=emb_method,
                    n_components=embedding_n_components,
                    target_col=nox_col if emb_method == "pls" else None,
                    emb_prefix=embedding_prefix
                )
                emb_cols = get_reduced_embedding_cols(df_reduced, method=emb_method)
                emb_method_desc = f"{emb_method.upper()} ({embedding_n_components} dims)"
                nox_controls.extend(emb_cols)
            else:
                df_reduced = df_nox
        else:
            df_reduced = df_nox
        
        print("\n" + "=" * 70)
        print(f"SATELLITE NOx ({emb_method.upper()}) — Detection Limit: {dl_label}")
        if emb_method_desc:
            print(f"  Controls: base + embeddings {emb_method_desc}")
        else:
            print(f"  Controls: base only (no embeddings)")
        if use_weights:
            print(f"  Weights: inverse-variance ({nox_weight_col})")
        print("=" * 70)
        
        results[f"satellite_{emb_method}"] = run_outcome_models(
            df_reduced, outcome_col=nox_col, treatment_col=treatment_col, # type: ignore
            controls=nox_controls if nox_controls else None,
            cluster_col=cluster_col,
            weight_col=nox_weight_col if use_weights else None,
            label=f"Satellite NOx ({emb_method.upper()}, DL {dl_label})"
        )
    
    return results


def run_dual_outcome_analysis(
    df: pd.DataFrame,
    treatment_col: str,
    controls: Optional[List[str]] = None,
    cluster_col: str = _DEFAULT_CLUSTER_COL,
    ets_col: str = "log_ets_co2",
    nox_col: str = "beirle_nox_kg_s",
    nox_dl_col: str = "above_dl_0_03",
    nox_weight_col: Optional[str] = "nox_weight",
    embedding_n_components: int = 10,
    embedding_prefix: str = "emb_"
) -> Dict[str, Dict]:
    """
    Run TWFE for both ETS CO2 and Satellite NOx outcomes.
    
    - ETS CO₂: Full sample, no embeddings
    - Satellite NOx: Detection-limit-filtered sample, with PCA and PLS embeddings
    
    IMPORTANT: NOx analysis ALWAYS applies detection limit filter. 
    Samples below detection limit are excluded.
    
    Parameters
    ----------
    nox_dl_col : str
        Detection limit column: 'above_dl_0_03' (permissive) or 'above_dl_0_11' (conservative)
    
    Returns
    -------
    Dict with keys: 'ets', 'satellite_pca', 'satellite_pls'
    """
    results = {}
    
    # ETS CO2 on full sample
    results["ets"] = run_ets_analysis(
        df, treatment_col=treatment_col,
        controls=controls, cluster_col=cluster_col,
        ets_col=ets_col
    )
    
    # Satellite NOx on DL-filtered sample (both PCA and PLS)
    nox_results = run_nox_analysis(
        df, treatment_col=treatment_col,
        controls=controls, cluster_col=cluster_col,
        nox_col=nox_col, nox_dl_col=nox_dl_col,
        nox_weight_col=nox_weight_col,
        embedding_n_components=embedding_n_components,
        embedding_prefix=embedding_prefix
    )
    results.update(nox_results)
    
    return results
