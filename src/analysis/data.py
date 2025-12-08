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
from typing import Optional, Tuple

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
DL_0_11_COL = "above_dl_0_11"
DL_0_03_COL = "above_dl_0_03"
DL_CONSERVATIVE_COL = DL_0_11_COL
DL_PERMISSIVE_COL = DL_0_03_COL
REL_ERR_STAT_LT_0_3_COL = "rel_err_stat_lt_0_3"
INTERFERED_20KM_COL = "interfered_20km"
URBANIZATION_DEGREE_COL = "urbanization_degree"
IN_URBAN_AREA_COL = "in_urban_area"


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
    # Note: urbanization variables are for heterogeneity analysis, not regression controls
    static_cols = [FAC_ID_COL, "lat", "lon", "country_code", 
                   URBANIZATION_DEGREE_COL, IN_URBAN_AREA_COL]
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
    
    # Merge satellite outcomes
    if include_satellite and beirle_path is not None:
        try:
            beirle = load_beirle_panel(beirle_path)
            # Keep only key columns from Beirle panel
            beirle_cols = [FAC_ID_COL, YEAR_COL, NOX_OUTCOME_COL, NOX_SE_COL,
                          "rel_err_total", "rel_err_stat", "rel_err_tau",
                          DL_CONSERVATIVE_COL, DL_PERMISSIVE_COL, "n_days_satellite"]
            beirle_cols = [c for c in beirle_cols if c in beirle.columns]
            
            panel = panel.merge(
                beirle[beirle_cols],
                on=[FAC_ID_COL, YEAR_COL],
                how="left"
            )
            n_with_sat = panel[NOX_OUTCOME_COL].notna().sum()
            print(f"Satellite outcomes: {n_with_sat} obs with valid NOx data")
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
    df[cohort_col] = df[id_col].map(first_treated).fillna(0).astype(int)
    
    # Summary
    n_ever = (df.groupby(id_col)[treated_col].max() > 0).sum()
    n_never = df[id_col].nunique() - n_ever
    print(f"Treatment: {n_ever} ever-treated, {n_never} never-treated")
    
    cohorts = df[df[cohort_col] > 0].groupby(id_col)[cohort_col].first().value_counts()
    print(f"Cohorts: {len(cohorts)}")
    
    return df


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
    facs_above = df[df[co2_col] >= min_co2_tco2][id_col].unique()
    df = df[df[id_col].isin(facs_above)]
    
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
        df = df[(df[time_col] >= year_range[0]) & (df[time_col] <= year_range[1])]
    
    # ETS emissions filter (only for ETS CO₂ analysis)
    if apply_ets_filter:
        df = apply_ets_emissions_filter(df, min_co2_tco2=min_co2_tco2, id_col=id_col)
    
    # Min years per facility
    years_per_fac = df.groupby(id_col)[time_col].nunique()
    keep_facs = years_per_fac[years_per_fac >= min_years].index
    df = df[df[id_col].isin(keep_facs)]
    
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
        - "significant": above_dl_0_11 AND rel_err_stat_lt_0_3
    exclude_interfered : bool
        If True, also exclude facilities with interfered_20km == True
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
        # Detection limit: >= 0.11 kg/s (standard European conditions)
        if DL_0_11_COL in df.columns:
            df = df[df[DL_0_11_COL] == True]
        # Statistical integration error < 30%
        if REL_ERR_STAT_LT_0_3_COL in df.columns:
            df = df[df[REL_ERR_STAT_LT_0_3_COL] == True]
    elif sample != "all":
        raise ValueError(f"Unknown sample: {sample}. Use 'all' or 'significant'.")
    
    # Optionally exclude interfered facilities
    if exclude_interfered and INTERFERED_20KM_COL in df.columns:
        df = df[df[INTERFERED_20KM_COL] == False]
    
    n_after = len(df)
    n_fac = df[FAC_ID_COL].nunique() if FAC_ID_COL in df.columns else "?"
    print(f"Satellite sample ({sample}, interfered_excl={exclude_interfered}): "
          f"{n_before} → {n_after} obs ({n_fac} facilities)")
    
    return df


def apply_satellite_filters(
    df: pd.DataFrame,
    detection_limit: str = "conservative",
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
    detection_limit : str
        "conservative" (0.11 kg/s) or "permissive" (0.03 kg/s)
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
    
    # Detection limit filter (using new column names)
    dl_col = DL_0_11_COL if detection_limit == "conservative" else DL_0_03_COL
    if dl_col in df.columns:
        df = df[df[dl_col] == True]
        print(f"Detection limit ({detection_limit}): {len(df)} obs retained")
    
    # Relative uncertainty filter
    if max_rel_err and "rel_err_total" in df.columns:
        df = df[df["rel_err_total"] <= max_rel_err]
        print(f"Max rel. error ({max_rel_err:.0%}): {len(df)} obs retained")
    
    # Minimum observation days
    if min_days and "n_days_satellite" in df.columns:
        df = df[df["n_days_satellite"] >= min_days]
        print(f"Min days ({min_days}): {len(df)} obs retained")
    
    n_after = len(df)
    print(f"Satellite filters: {n_before} → {n_after} obs")
    
    return df


def get_intersection_sample(
    df: pd.DataFrame,
    ets_col: str = ETS_CO2_COL,
    nox_col: str = NOX_OUTCOME_COL,
    satellite_sample: str = "significant",
    exclude_interfered: bool = False
) -> pd.DataFrame:
    """
    Get sample with both valid ETS and satellite outcomes.
    
    Useful for comparing treatment effects across outcomes on same sample.
    Uses the new boolean-flag-based filtering via get_satellite_sample().
    
    Parameters
    ----------
    df : DataFrame
        Panel with both ETS and satellite outcomes
    ets_col : str
        ETS outcome column
    nox_col : str
        Satellite NOx outcome column
    satellite_sample : str
        "all" or "significant" (passed to get_satellite_sample)
    exclude_interfered : bool
        If True, exclude interfered facilities
    
    Returns
    -------
    Filtered DataFrame with both outcomes valid
    """
    df = df.copy()
    
    # Require ETS outcome
    df = df.dropna(subset=[ets_col])
    
    # Apply satellite sample filter
    df = get_satellite_sample(df, sample=satellite_sample, exclude_interfered=exclude_interfered)
    
    n_fac = df[FAC_ID_COL].nunique()
    print(f"Intersection sample: {len(df)} obs, {n_fac} facilities")
    
    return df


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
        (DL_0_11_COL, "above_dl_0_11 (≥0.11 kg/s)"),
        (DL_0_03_COL, "above_dl_0_03 (≥0.03 kg/s)"),
        (REL_ERR_STAT_LT_0_3_COL, "rel_err_stat < 30%"),
        (INTERFERED_20KM_COL, "interfered_20km"),
        (IN_URBAN_AREA_COL, "in_urban_area (SMOD ≥21)"),
    ]:
        if col in sat_df.columns:
            n_true = sat_df[col].sum()
            pct = 100 * n_true / n_total if n_total > 0 else 0
            summary.append({"flag": label, "n_true": n_true, "pct": f"{pct:.1f}%", "n_total": n_total})
    
    summary_df = pd.DataFrame(summary)
    print(f"\nSatellite significance flags ({n_total} obs, {n_fac} facilities):")
    print(summary_df.to_string(index=False))
    
    return summary_df
