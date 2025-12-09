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
    
    # Extract results
    coef = model.coef().get(treatment_col, np.nan)
    se = model.se().get(treatment_col, np.nan)
    pval = model.pvalue().get(treatment_col, np.nan)
    
    results = {
        "spec_name": spec_name,
        "coefficient": coef,
        "se": se,
        "pvalue": pval,
        "ci_lower": coef - 1.96 * se, # type: ignore
        "ci_upper": coef + 1.96 * se, # type: ignore
        "n_obs": model.nobs, # type: ignore
        "r2": model.r2, # type: ignore
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


def run_dual_outcome_analysis(
    df: pd.DataFrame,
    treatment_col: str,
    controls: Optional[List[str]] = None,
    cluster_col: str = _DEFAULT_CLUSTER_COL,
    ets_col: str = "log_ets_co2",
    nox_col: str = "beirle_nox_kg_s",
    nox_se_col: str = "beirle_nox_kg_s_se",
    nox_days_control: bool = False,
    nox_days_col: str = "n_days_satellite",
    nox_embeddings: bool = True,
    nox_weight_col: Optional[str] = "nox_weight",
    embedding_reduction: Optional[str] = None,
    embedding_n_components: int = 10,
    embedding_prefix: str = "emb_"
) -> Dict[str, Dict]:
    """
    Run TWFE for both ETS CO2 and Satellite NOx outcomes.
    
    Uses single specification: Facility + Region×Year FE with clustered SEs.
    
    Parameters
    ----------
    controls : list
        Base controls (e.g., capacity, fuel shares) applied to BOTH outcomes.
    nox_days_control : bool
        If True, add n_days_satellite and nox_se_col as controls for NOx outcome.
    nox_days_col : str
        Column name for satellite observation day count.
    nox_embeddings : bool
        If True (default), add AlphaEarth embeddings as controls for NOx
        outcome ONLY. Embeddings control for satellite retrieval heterogeneity
        (terrain, land use, climate) but are irrelevant for ETS CO₂ (administrative).
    nox_weight_col : str, optional
        Column containing inverse-variance weights for NOx outcome (default: 'nox_weight').
        Set to None to disable weighting.
    embedding_reduction : str, optional
        Dimensionality reduction for embeddings: 'pca' (unsupervised) or 'pls' 
        (facility-level supervised). If None (default), uses raw 64-dim embeddings.
        PLS is trained on facility-level mean NOx to prevent outcome snooping.
        See Chernozhukov et al. (2018) on regularization bias.
    embedding_n_components : int
        Number of components for reduced embeddings (default: 10).
    embedding_prefix : str
        Prefix for raw embedding columns (default: 'emb_').
    """
    results = {}
    
    # ==========================================================================
    # ETS CO2 (ground truth) - NO EMBEDDINGS
    # Geographic confounders absorbed by Facility FE (time-invariant) and
    # Region×Year FE (time-varying). Embeddings would control for nothing relevant.
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ETS VERIFIED CO2 EMISSIONS (GROUND TRUTH)")
    print("  Controls: base only (embeddings NOT used - absorbed by FE)")
    print("=" * 70)
    results["ets"] = run_outcome_models(
        df, outcome_col=ets_col, treatment_col=treatment_col,
        controls=controls, cluster_col=cluster_col,
        label="ETS CO2 (tCO2/yr, log)"
    )
    
    # ==========================================================================
    # Satellite NOx - WITH EMBEDDINGS (for retrieval context)
    # Embeddings control for systematic measurement heterogeneity:
    # - Terrain affects AMF corrections
    # - Land use affects urban background NO₂
    # - Climate affects wind patterns and NOx lifetime
    # ==========================================================================
    nox_controls = list(controls) if controls else []
    
    # Add observation coverage controls
    if nox_days_control:
        if nox_days_col in df.columns:
            nox_controls.append(nox_days_col)
        if nox_se_col in df.columns:
            nox_controls.append(nox_se_col)
    
    # Add AlphaEarth embeddings for satellite retrieval context
    n_emb_added = 0
    emb_method_desc = ""
    if nox_embeddings:
        raw_emb_cols = [c for c in df.columns if c.startswith(embedding_prefix)]
        
        if raw_emb_cols:
            # Check how many NOx obs have valid embeddings
            nox_valid = df[nox_col].notna()
            emb_valid = df[raw_emb_cols[0]].notna()
            n_both = (nox_valid & emb_valid).sum()
            
            if n_both > 0:
                if embedding_reduction is not None:
                    # Apply dimensionality reduction
                    from embedding_reduction import reduce_embeddings, get_reduced_embedding_cols
                    
                    df = reduce_embeddings(
                        df, 
                        method=embedding_reduction,
                        n_components=embedding_n_components,
                        target_col=nox_col if embedding_reduction == "pls" else None,
                        emb_prefix=embedding_prefix
                    )
                    emb_cols = get_reduced_embedding_cols(df, method=embedding_reduction)
                    emb_method_desc = f"{embedding_reduction.upper()} ({embedding_n_components} dims)"
                else:
                    # Use raw embeddings
                    emb_cols = raw_emb_cols
                    emb_method_desc = f"raw ({len(raw_emb_cols)} dims)"
                
                nox_controls.extend(emb_cols)
                n_emb_added = len(emb_cols)
    
    # Check if weights are available
    use_weights = nox_weight_col and nox_weight_col in df.columns
    
    print("\n" + "=" * 70)
    print("SATELLITE NOx EMISSION PROXY (BEIRLE-STYLE)")
    if emb_method_desc:
        print(f"  Controls: base + embeddings {emb_method_desc}")
    else:
        print(f"  Controls: base only (no embeddings)")
    if nox_days_control:
        print(f"  + observation coverage controls ({nox_days_col})")
    if use_weights:
        print(f"  Weights: inverse-variance ({nox_weight_col})")
    print("=" * 70)
    
    results["satellite"] = run_outcome_models(
        df, outcome_col=nox_col, treatment_col=treatment_col,
        controls=nox_controls if nox_controls else None,
        cluster_col=cluster_col,
        weight_col=nox_weight_col if use_weights else None,
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
