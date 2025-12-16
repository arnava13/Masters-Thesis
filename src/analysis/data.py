"""
data.py
=======
Data loading and preprocessing for analysis panel.

Loads output from Data Pipeline.ipynb and constructs treatment variables.

Dual Outcomes:
- ETS CO₂: Verified annual emissions from EU ETS registry (ground-truth)
- Satellite NOx: Beirle-style flux-divergence proxy (noisy, with uncertainty)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict

# Default paths (pass paths explicitly from notebook)
_DEFAULT_OUT_DIR = Path("out")

# Default column names (can be overridden via function parameters)
FAC_ID_COL = "idx"
YEAR_COL = "year"
ALLOC_RATIO_COL = "eu_alloc_ratio"
ETS_CO2_COL = "eu_verified_tco2"
LOG_ETS_CO2_COL = "log_ets_co2"
MIN_VERIFIED_CO2_TCO2 = 100_000
NOX_OUTCOME_COL = "beirle_nox_kg_s"
NOX_SE_COL = "beirle_nox_kg_s_se"
REPORTED_NOX_COL = "reported_nox_tonnes"
LOG_REPORTED_NOX_COL = "log_reported_nox"
ABOVE_DL_COL = "above_dl"
REL_ERR_STAT_LT_0_3_COL = "rel_err_stat_lt_0_3"
N_DAYS_SATELLITE_COL = "n_days_satellite"
INTERFERED_COL = "interfered_5km"
URBANIZATION_DEGREE_COL = "urbanization_degree"
IN_URBAN_AREA_COL = "in_urban_area"
IS_ELECTRICITY_COL = "is_electricity"  # EU ETS activity codes 1 or 20 (combustion)


# =============================================================================
# Data Loading
# =============================================================================

def load_facilities_static(path: Path) -> pd.DataFrame:
    """Load static facility attributes (idx, lat, lon, country_code, etc.)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Static file not found: {path}\n"
            "Run Data Pipeline.ipynb first."
        )
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} facilities from {path.name}")
    return df


def load_facilities_yearly(path: Path) -> pd.DataFrame:
    """Load yearly panel (idx, year, fuel shares, ETS, outcomes)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Yearly file not found: {path}\n"
            "Run Data Pipeline.ipynb first."
        )
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} obs from {path.name}")
    return df


def load_beirle_panel(path: Path) -> pd.DataFrame:
    """
    Load Beirle-style satellite NOx emission panel.
    
    Contains: beirle_nox_kg_s, uncertainties, detection limit flags.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Beirle panel not found: {path}\n"
            "Run Data Pipeline.ipynb (satellite section) first."
        )
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} satellite obs from {path.name}")
    return df


def load_analysis_panel(
    static_path: Path,
    yearly_path: Path,
    beirle_path: Optional[Path] = None,
    include_satellite: bool = True
) -> pd.DataFrame:
    """
    Load and merge static + yearly + satellite data into analysis panel.
    
    Parameters
    ----------
    static_path : Path, optional
        Path to facilities_static.parquet
    yearly_path : Path, optional
        Path to facilities_yearly.parquet
    beirle_path : Path, optional
        Path to beirle_panel.parquet
    include_satellite : bool
        If True, merge Beirle satellite outcomes
    
    Returns
    -------
    DataFrame with facility info, ETS outcomes, and optionally satellite outcomes
    """
    static = load_facilities_static(static_path)
    yearly = load_facilities_yearly(yearly_path)
    
    # Merge static info into yearly panel
    # Note: urbanization and interference variables are for heterogeneity analysis, not regression controls
    static_cols = [FAC_ID_COL, "lat", "lon", "country_code", "nuts2_region", "pypsa_cluster",
                   "is_electricity", URBANIZATION_DEGREE_COL, IN_URBAN_AREA_COL, INTERFERED_COL]
    static_cols = [c for c in static_cols if c in static.columns]
    
    panel = yearly.merge(
        static[static_cols],
        on=FAC_ID_COL,
        how="left"
    )
    
    # Create log-transformed ETS CO₂ outcome
    if ETS_CO2_COL in panel.columns:
        # Add small offset to handle zeros
        panel[LOG_ETS_CO2_COL] = np.log(panel[ETS_CO2_COL] + 1)
    
    # Create log-transformed reported NOx outcome
    if REPORTED_NOX_COL in panel.columns:
        # Add small offset to handle zeros; NaN preserved
        panel[LOG_REPORTED_NOX_COL] = np.log(panel[REPORTED_NOX_COL] + 1)
    
    # Merge satellite outcomes (includes AlphaEarth embeddings for NOx retrieval context)
    # NOTE: Embeddings are stored in beirle_panel, NOT facilities_yearly, because they
    # are controls for the satellite measurement process (terrain, land use, climate)
    # and do NOT affect ETS CO₂ (administrative mass balance reporting).
    if include_satellite and beirle_path is not None:
        try:
            beirle = load_beirle_panel(beirle_path)
            # Keep key columns from Beirle panel (including embeddings if present)
            beirle_cols = [FAC_ID_COL, YEAR_COL, NOX_OUTCOME_COL, NOX_SE_COL,
                          "rel_err_total", "rel_err_stat", "rel_err_tau",
                          ABOVE_DL_COL, N_DAYS_SATELLITE_COL]
            # Add embedding columns (emb_* pattern)
            emb_cols = [c for c in beirle.columns if c.startswith("emb_")]
            beirle_cols.extend(emb_cols)
            beirle_cols = [c for c in beirle_cols if c in beirle.columns]
            
            panel = panel.merge(
                beirle[beirle_cols],
                on=[FAC_ID_COL, YEAR_COL],
                how="left"
            )
            n_with_sat = panel[NOX_OUTCOME_COL].notna().sum()
            n_with_emb = panel[emb_cols[0]].notna().sum() if emb_cols else 0
            print(f"Satellite outcomes: {n_with_sat} obs with valid NOx data")
            if emb_cols:
                print(f"AlphaEarth embeddings: {len(emb_cols)} dims, {n_with_emb} obs (NOx analysis only)")
        except FileNotFoundError:
            print("Warning: Beirle panel not found, satellite outcomes unavailable")
    
    n_fac = panel[FAC_ID_COL].nunique()
    years = f"{panel[YEAR_COL].min()}-{panel[YEAR_COL].max()}"
    print(f"Panel: {len(panel)} obs, {n_fac} facilities, {years}")
    
    return panel


# =============================================================================
# Treatment Variable Construction
# =============================================================================

def build_treatment_variables(
    df: pd.DataFrame,
    treatment_col: str = ALLOC_RATIO_COL,
    threshold: float = 1.0,
    treated_col: str = "treated",
    cohort_col: str = "cohort",
    id_col: str = FAC_ID_COL,
    time_col: str = YEAR_COL
) -> pd.DataFrame:
    """
    Build treatment variables for continuous and discrete specifications.
    
    Parameters
    ----------
    df : DataFrame
        Panel data with treatment_col
    treatment_col : str
        Continuous treatment (e.g., eu_alloc_ratio)
    threshold : float
        Threshold for discrete treatment (treated if < threshold)
    treated_col : str
        Output column for binary treatment
    cohort_col : str
        Output column for first treatment year
    
    Returns
    -------
    DataFrame with treatment variables added
    """
    df = df.copy()
    
    # Binary treatment: 1 if below threshold
    df[treated_col] = (df[treatment_col] < threshold).astype(int)
    
    # Cohort: first year treated (0 = never treated)
    first_treated = (
        df[df[treated_col] == 1]
        .groupby(id_col)[time_col]
        .min()
    )
    df[cohort_col] = df[id_col].map(first_treated).fillna(0).astype(int) # type: ignore
    
    # Summary
    n_ever = (df.groupby(id_col)[treated_col].max() > 0).sum() # type: ignore
    n_never = df[id_col].nunique() - n_ever
    print(f"Treatment: {n_ever} ever-treated, {n_never} never-treated")
    
    cohorts = df[df[cohort_col] > 0].groupby(id_col)[cohort_col].first().value_counts()
    print(f"Cohorts: {len(cohorts)}")
    
    return df


def identify_treatment_reversers(
    df: pd.DataFrame,
    treatment_col: str = ALLOC_RATIO_COL,
    threshold: float = 1.0,
    id_col: str = FAC_ID_COL,
    time_col: str = YEAR_COL
) -> pd.DataFrame:
    """
    Identify facilities with treatment reversals.
    
    A reversal occurs when a facility's treatment status changes from treated
    to untreated (or vice versa) between consecutive periods. Standard CS-DiD
    assumes absorbing treatment (once treated, always treated), so reversers
    violate this assumption.
    
    Parameters
    ----------
    df : DataFrame
        Panel data
    treatment_col : str
        Treatment intensity column (e.g., eu_alloc_ratio)
    threshold : float
        Threshold for binary treatment (treated if < threshold)
    
    Returns
    -------
    DataFrame with columns: idx, n_reversals, is_reverser
    """
    df = df.copy()
    df["_treated"] = (df[treatment_col] < threshold).astype(int)
    
    # Sort by facility and time
    df = df.sort_values([id_col, time_col])
    
    # Count reversals per facility
    def count_reversals(seq):
        reversals = 0
        seq_list = list(seq)
        for i in range(1, len(seq_list)):
            if seq_list[i] != seq_list[i-1]:
                reversals += 1
        return reversals
    
    reversal_counts = (
        df.groupby(id_col)["_treated"]
        .apply(count_reversals)
        .reset_index()
        .rename(columns={"_treated": "n_reversals"})
    )
    reversal_counts["is_reverser"] = reversal_counts["n_reversals"] > 0
    
    return reversal_counts


def get_absorbing_treatment_sample(
    df: pd.DataFrame,
    treatment_col: str = ALLOC_RATIO_COL,
    threshold: float = 1.0,
    id_col: str = FAC_ID_COL,
    time_col: str = YEAR_COL
) -> pd.DataFrame:
    """
    Filter to facilities with absorbing treatment (no reversals).
    
    Required for valid Callaway-Sant'Anna estimation, which assumes once
    treated, always treated. Facilities that switch in/out of treatment
    are excluded.
    
    Parameters
    ----------
    df : DataFrame
        Panel data
    treatment_col : str
        Treatment intensity column
    threshold : float
        Threshold for binary treatment
    
    Returns
    -------
    Filtered DataFrame with only non-reversing facilities
    """
    reversal_info = identify_treatment_reversers(
        df, treatment_col, threshold, id_col, time_col
    )
    
    n_total = df[id_col].nunique()
    non_reversers = reversal_info[~reversal_info["is_reverser"]][id_col]
    n_keep = len(non_reversers)
    n_drop = n_total - n_keep
    
    df_filtered = df[df[id_col].isin(non_reversers)].copy() # type: ignore
    
    print(f"\nAbsorbing treatment filter (CS-DiD requirement):")
    print(f"  Total facilities: {n_total}")
    print(f"  Reversers dropped: {n_drop} ({100*n_drop/n_total:.1f}%)")
    print(f"  Non-reversers kept: {n_keep} ({100*n_keep/n_total:.1f}%)")
    print(f"  Observations: {len(df)} → {len(df_filtered)}")
    
    return df_filtered # type: ignore


# =============================================================================
# Sample Filters
# =============================================================================

def apply_ets_emissions_filter(
    df: pd.DataFrame,
    min_co2_tco2: float = MIN_VERIFIED_CO2_TCO2,
    id_col: str = FAC_ID_COL,
    co2_col: str = ETS_CO2_COL,
) -> pd.DataFrame:
    """
    Apply emissions filter for ETS CO₂ analysis.
    
    Keeps facilities with at least one year above the threshold. This filter
    is specific to the ETS CO₂ outcome and should NOT be applied to the
    satellite NOx panel (which has its own detection limits).
    
    Parameters
    ----------
    df : DataFrame
        Panel data with ETS CO₂ column
    min_co2_tco2 : float
        Minimum verified CO₂ threshold in tCO₂/yr (default: 100,000)
    id_col : str
        Facility ID column
    co2_col : str
        ETS CO₂ column name
    
    Returns
    -------
    Filtered DataFrame (facilities with ≥1 year above threshold)
    """
    if co2_col not in df.columns:
        print(f"Warning: {co2_col} not in DataFrame, skipping emissions filter")
        return df
    
    n_fac_before = df[id_col].nunique()
    n_before = len(df)
    
    # Find facilities with at least one year above threshold
    facs_above = df[df[co2_col] >= min_co2_tco2][id_col].unique() # type: ignore
    df = df[df[id_col].isin(facs_above)] # type: ignore
    
    n_fac_after = df[id_col].nunique()
    n_after = len(df)
    
    print(f"ETS emissions filter (≥{min_co2_tco2/1000:.0f} ktCO₂/yr): "
          f"{n_fac_before} → {n_fac_after} facilities, {n_before} → {n_after} obs")
    
    return df


def apply_sample_filters(
    df: pd.DataFrame,
    min_years: int = 3,
    year_range: Optional[Tuple[int, int]] = None,
    require_outcome: bool = True,
    outcome_col: Optional[str] = None,
    apply_ets_filter: bool = False,
    min_co2_tco2: float = MIN_VERIFIED_CO2_TCO2,
    id_col: str = FAC_ID_COL,
    time_col: str = YEAR_COL
) -> pd.DataFrame:
    """
    Apply sample restrictions.
    
    Parameters
    ----------
    df : DataFrame
        Panel data
    min_years : int
        Minimum years per facility (for balanced-ish panel)
    year_range : tuple, optional
        (start_year, end_year) to keep
    require_outcome : bool
        If True, drop rows with missing outcome
    outcome_col : str, optional
        Outcome column to check for missingness
    apply_ets_filter : bool
        If True, apply ETS emissions filter (≥min_co2_tco2 in any year).
        Only use for ETS CO₂ analysis, NOT for satellite NOx.
    min_co2_tco2 : float
        Minimum CO₂ threshold for ETS filter (default: 100,000 tCO₂/yr)
    
    Returns
    -------
    Filtered DataFrame
    """
    n_before = len(df)
    
    # Year range
    if year_range:
        df = df[(df[time_col] >= year_range[0]) & (df[time_col] <= year_range[1])] # type: ignore
    
    # ETS emissions filter (only for ETS CO₂ analysis)
    if apply_ets_filter:
        df = apply_ets_emissions_filter(df, min_co2_tco2=min_co2_tco2, id_col=id_col)
    
    # Min years per facility
    years_per_fac = df.groupby(id_col)[time_col].nunique()
    keep_facs = years_per_fac[years_per_fac >= min_years].index # type: ignore
    df = df[df[id_col].isin(keep_facs)] # type: ignore
    
    # Require outcome
    if require_outcome and outcome_col:
        df = df.dropna(subset=[outcome_col])
    
    n_after = len(df)
    n_fac = df[id_col].nunique()
    print(f"Filters: {n_before} → {n_after} obs ({n_fac} facilities)")
    
    return df


# =============================================================================
# Satellite Sample Definitions
# =============================================================================

# Satellite NOx is treated as a **noisy proxy** outcome. We follow Beirle et al.
# for quantification but apply simplified significance filters (detection limit,
# integration error, interference flag). ETS verified CO₂ remains the primary outcome.

def get_satellite_sample(
    df: pd.DataFrame,
    sample: str = "all",
    exclude_interfered: bool = False
) -> pd.DataFrame:
    """
    Get satellite sample using boolean flags (no hard-coded numeric cuts).
    
    This function implements the principle of skipping Beirle's automatic
    point-source identification (we have known ETS/LCP locations) but applying
    their significance logic via pre-computed boolean flags.
    
    Parameters
    ----------
    df : DataFrame
        Panel with satellite outcomes and significance flags
    sample : str
        Sample definition:
        - "all": All facilities with non-missing satellite outcome
        - "significant": above_dl_conservative AND rel_err_stat_lt_0_3
    exclude_interfered : bool
        If True, also exclude facilities with interfered_{n}km == True
        (where satellite outcome reflects cluster-level emissions)
    
    Returns
    -------
    Filtered DataFrame
    """
    df = df.copy()
    n_before = len(df)
    
    # Require non-missing satellite outcome
    if NOX_OUTCOME_COL in df.columns:
        df = df.dropna(subset=[NOX_OUTCOME_COL])
    
    # Apply significance filters based on sample definition
    if sample == "significant":
        # Detection limit: >= 0.01 kg/s (permissive threshold for statistical power)
        if ABOVE_DL_COL in df.columns:
            df = df[df[ABOVE_DL_COL] == True] # type: ignore
        # Statistical integration error < 30%
        if REL_ERR_STAT_LT_0_3_COL in df.columns:
            df = df[df[REL_ERR_STAT_LT_0_3_COL] == True] # type: ignore
    elif sample != "all":
        raise ValueError(f"Unknown sample: {sample}. Use 'all' or 'significant'.")
    
    # Optionally exclude interfered facilities
    if exclude_interfered and INTERFERED_COL in df.columns:
        df = df[df[INTERFERED_COL] == False] # type: ignore
    
    n_after = len(df)
    n_fac = df[FAC_ID_COL].nunique() if FAC_ID_COL in df.columns else "?"
    print(f"Satellite sample ({sample}, interfered_excl={exclude_interfered}): "
          f"{n_before} → {n_after} obs ({n_fac} facilities)")
    
    return df


def apply_satellite_filters(
    df: pd.DataFrame,
    max_rel_err: Optional[float] = None,
    min_days: int = 30
) -> pd.DataFrame:
    """
    Apply satellite-specific filters for NOx outcome analysis.
    
    DEPRECATED: Prefer get_satellite_sample() which uses boolean flags.
    This function is kept for backward compatibility.
    
    Parameters
    ----------
    df : DataFrame
        Panel with satellite outcomes
    max_rel_err : float, optional
        Maximum total relative uncertainty (e.g., 0.5 for 50%)
    min_days : int
        Minimum days with valid satellite observations
    
    Returns
    -------
    Filtered DataFrame
    """
    import warnings
    warnings.warn(
        "apply_satellite_filters() is deprecated. Use get_satellite_sample() "
        "which uses boolean flags instead of hard-coded numeric cuts.",
        DeprecationWarning
    )
    
    n_before = len(df)
    df = df.copy()
    
    # Detection limit filter (single threshold: 0.01 kg/s)
    if ABOVE_DL_COL in df.columns:
        df = df[df[ABOVE_DL_COL] == True] # type: ignore
        print(f"Detection limit (≥0.01 kg/s): {len(df)} obs retained")
    
    # Relative uncertainty filter
    if max_rel_err and "rel_err_total" in df.columns:
        df = df[df["rel_err_total"] <= max_rel_err] # type: ignore
        print(f"Max rel. error ({max_rel_err:.0%}): {len(df)} obs retained")
    
    # Minimum observation days
    if min_days and "n_days_satellite" in df.columns:
        df = df[df["n_days_satellite"] >= min_days] # type: ignore
        print(f"Min days ({min_days}): {len(df)} obs retained")
    
    n_after = len(df)
    print(f"Satellite filters: {n_before} → {n_after} obs")
    
    return df


def get_satellite_sample_filtered(
    df: pd.DataFrame,
    exclude_interfered: bool = False
) -> pd.DataFrame:
    """
    Get satellite sample with valid outcomes above detection limit.
    
    Uses single detection limit of 0.01 kg/s (chosen for statistical power,
    acknowledging this is well below Beirle's validated thresholds).
    
    Parameters
    ----------
    df : DataFrame
        Panel with both ETS and satellite outcomes
    ets_col : str
        ETS outcome column
    exclude_interfered : bool
        If True, exclude interfered facilities
    
    Returns
    -------
    Filtered DataFrame with valid satellite observations above DL
    """
    # Base: require ETS outcome and non-missing satellite
    base_df = df.copy()
    base_df = base_df.dropna(subset=[NOX_OUTCOME_COL])
    
    print("\n" + "=" * 60)
    print("SATELLITE SAMPLE (ETS + Satellite)")
    print("=" * 60)
    
    # Detection limit: ≥ 0.01 kg/s
    if ABOVE_DL_COL in base_df.columns:
        result_df = base_df[base_df[ABOVE_DL_COL] == True].copy()
    else:
        result_df = base_df.copy()  # If no flag, use all
    
    if exclude_interfered and INTERFERED_COL in result_df.columns:
        result_df = result_df[result_df[INTERFERED_COL] == False]
    
    n_obs = len(result_df)
    n_fac = result_df[FAC_ID_COL].nunique() # type: ignore
    print(f"  DL ≥ 0.01 kg/s: {n_obs} obs, {n_fac} facilities")
    
    return result_df # type: ignore


def summarize_satellite_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize satellite significance flags for diagnostics.
    
    Returns a DataFrame with counts and percentages for each flag.
    """
    # Require satellite outcome
    sat_df = df.dropna(subset=[NOX_OUTCOME_COL]) if NOX_OUTCOME_COL in df.columns else df
    n_total = len(sat_df)
    n_fac = sat_df[FAC_ID_COL].nunique() if FAC_ID_COL in sat_df.columns else None
    
    summary = []
    for col, label in [
        (ABOVE_DL_COL, "above_dl (≥0.01 kg/s)"),
        (REL_ERR_STAT_LT_0_3_COL, "rel_err_stat < 30%"),
        (INTERFERED_COL, "interfered_5km"),
        (IN_URBAN_AREA_COL, "in_urban_area (SMOD ≥22)"),
    ]:
        if col in sat_df.columns:
            n_true = sat_df[col].sum()
            pct = 100 * n_true / n_total if n_total > 0 else 0
            summary.append({"flag": label, "n_true": n_true, "pct": f"{pct:.1f}%", "n_total": n_total})
    
    summary_df = pd.DataFrame(summary)
    print(f"\nSatellite significance flags ({n_total} obs, {n_fac} facilities):")
    print(summary_df.to_string(index=False))
    
    return summary_df
