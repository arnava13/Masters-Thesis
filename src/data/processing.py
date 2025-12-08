"""
Data processing utilities for LCP/ETS pipeline.

Contains:
- fuel_group_map: Maps raw fuel types to standardized fuel groups
- normalize_pyeutl_id: Normalizes pyeutl installation IDs
- normalize_eureg_id: Normalizes EU Registry ETS identifiers
- build_eu_ets_yearly_from_zip: Builds yearly ETS metrics from zip file
"""

import re
import pandas as pd
import numpy as np
from pyeutl.ziploader import get_installations, get_compliance  # type: ignore


def fuel_group_map(x: object) -> str:
    """Map raw fuel type to standardized fuel group."""
    s = str(x).strip().upper()
    if s in {'NATURALGAS', 'NG', 'GAS'}:
        return 'Gas'
    if s in {'COAL', 'PC', 'COW', 'COL', 'BIT', 'SUB', 'ANT', 'LIGNITE', 'LIG', 'WC', 'RC'}:
        return 'Coal'
    if s in {'LIQUIDFUELS', 'DFO', 'RFO', 'KER', 'JF', 'OIL'}:
        return 'Oil'
    if s in {'OTHERGASES', 'OBG'}:
        return 'Other gas'
    if s in {'OTHERSOLIDFUELS', 'OTHER SOLID FUELS', 'OSF'}:
        return 'Other solid'
    if s in {'BIOMASS', 'WDL', 'WDS', 'AB'}:
        return 'Biomass'
    if s in {'PEAT'}:
        return 'Peat'
    if s in {'NUC', 'NUCLEAR'}:
        return 'Nuclear'
    if s in {'SUN', 'SOLAR'}:
        return 'Solar'
    if s in {'WND', 'WIND'}:
        return 'Wind'
    if s in {'WAT', 'HYDRO'}:
        return 'Hydro'
    if s in {'GEO', 'GEOTHERMAL'}:
        return 'Geothermal'
    if s in {'MSW', 'LFG', 'TDF'}:
        return 'Waste'
    return 'Other/Unknown'


def normalize_pyeutl_id(x) -> str | None:
    """Normalize pyeutl format: 'AT_200165' -> 'AT_200165'."""
    if pd.isna(x):
        return None
    parts = str(x).split("_")
    if len(parts) >= 2:
        num = parts[-1].lstrip('0') or '0'
        return f"{parts[0]}_{num}"
    return None


def normalize_eureg_id(x) -> str | None:
    """Normalize EU Registry format: 'FR000000000210535' or 'FR-new-07101261' -> 'FR_210535' or 'FR_7101261'."""
    if pd.isna(x) or x == "":
        return None
    s = str(x)
    m = re.match(r'^([A-Z]{2})', s)
    country = m.group(1) if m else None
    nums = re.findall(r'\d+', s)
    if nums and country:
        num = max(nums, key=len).lstrip('0') or '0'
        return f"{country}_{num}"
    return None


def build_eu_ets_yearly_from_zip(
    fn_zip: str,
    fac_static,
    fac_id_col: str = 'idx',
    start_year: int = 2018,
    end_year: int = 2023,
    min_verified_co2_tco2: float = 100_000,
    alloc_ratio_min: float = 0.01,
    alloc_ratio_max: float = 20.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Build yearly EU ETS metrics aggregated at the facility level.
    Uses normalized ID matching between EU Registry ETS IDs and pyeutl installation IDs.
    
    Args:
        fn_zip: Path to EU ETS zip file
        fac_static: facilities_static DataFrame with fac_id_col and ets_id (list) columns
        fac_id_col: Name of facility ID column (default: 'idx')
        start_year: Start year for filtering (default: 2018)
        end_year: End year for filtering (default: 2023)
        min_verified_co2_tco2: Minimum verified CO2 threshold in tCO2 (default: 100,000)
        alloc_ratio_min: Minimum allocation ratio to keep (default: 0.01)
        alloc_ratio_max: Maximum allocation ratio to keep (default: 20.0)
    
    Returns:
        Tuple of (yearly_panel DataFrame, matched_ets_ids dict mapping facility idx -> list of matched NORMALIZED ETS IDs)
    """
    df_inst = get_installations(fn_zip)
    df_comp = get_compliance(fn_zip, df_installation=df_inst)
    
    print(f"Total ETS installations in pyeutl: {df_comp['installation_id'].nunique():,}")

    # Include relevant columns - use allocatedTotal (not just allocatedFree)
    need = ["installation_id", "name", "year", 
            "allocatedTotal", "verified", "surrendered",
            "country_id"]
    keep = [c for c in need if c in df_comp.columns]
    c = df_comp[keep].copy()

    # Normalize pyeutl IDs
    c["installation_id_norm"] = c["installation_id"].apply(normalize_pyeutl_id)

    # Rename fields
    c = c.rename(columns={
        "country_id": "country_code",
        "name": "ets_name",
        "allocatedTotal": "eu_alloc_total_tco2",
        "verified": "eu_verified_tco2",
        "surrendered": "eu_surrendered_tco2",
    })

    # Coerce numerics
    for col in ["year", "eu_alloc_total_tco2", "eu_verified_tco2", "eu_surrendered_tco2"]:
        if col in c.columns:
            c[col] = pd.to_numeric(c[col], errors="coerce")

    # --- ID-based matching using normalized IDs ---
    fac = fac_static[[fac_id_col, "name", "ets_id"]].copy()
    fac = fac.rename(columns={"name": "fac_name"}) # type: ignore
    
    # Explode and normalize EU Registry IDs (ets_id is a list per facility)
    fac_exploded = fac.explode("ets_id").dropna(subset=["ets_id"])
    fac_exploded = fac_exploded[fac_exploded["ets_id"] != ""]
    fac_exploded["ets_id_norm"] = fac_exploded["ets_id"].apply(normalize_eureg_id) # type: ignore
    fac_exploded = fac_exploded.dropna(subset=["ets_id_norm"]) # type: ignore
    
    print(f"Facilities with valid ETS IDs: {fac_exploded[fac_id_col].nunique():,} / {len(fac):,}")
    print(f"Total normalized ETS ID links: {len(fac_exploded):,}")
    
    # Join on normalized IDs
    matched = c.merge(
        fac_exploded[[fac_id_col, "ets_id", "ets_id_norm", "fac_name"]],
        left_on="installation_id_norm",
        right_on="ets_id_norm",
        how="inner"
    )
    
    matched_ets = matched["installation_id"].nunique()
    matched_fac = matched[fac_id_col].nunique()
    print(f"Matched ETS installations: {matched_ets:,}")
    print(f"Matched facilities: {matched_fac:,} ({100*matched_fac/len(fac):.1f}%)")
    
    if matched.empty:
        print("No matches found!")
        return pd.DataFrame(), {}
    
    # Track which normalised ETS IDs actually matched (for updating facilities_static)
    matched_ets_ids_per_fac = (
        matched.groupby(fac_id_col)["ets_id_norm"]
        .apply(lambda x: list(x.unique()))
        .to_dict()
    )
    
    # Deduplicate to ensure each installation is counted once per 
    # facility-year, as multiple ETS IDs may normalise to the same one
    n_before_dedup = len(matched)
    matched = matched.drop_duplicates(subset=[fac_id_col, "installation_id", "year"])
    n_after_dedup = len(matched)
    if n_before_dedup > n_after_dedup:
        print(f"Deduplicated (fac, installation, year): {n_before_dedup:,} → {n_after_dedup:,} rows")
    
    # Check for multiple installations per facility
    inst_per_fac = matched.groupby(fac_id_col)["installation_id"].nunique()
    multi = inst_per_fac[inst_per_fac > 1]
    if len(multi) > 0:
        print(f"Facilities with multiple ETS installations: {len(multi)} (max: {multi.max()})")

    # Aggregate to facility × year (now safe - each installation counted once)
    out = (
        matched.groupby([fac_id_col, "year"], as_index=False)
        .agg({
            "eu_verified_tco2": "sum",
            "eu_alloc_total_tco2": "sum",
            "eu_surrendered_tco2": "sum",
            "installation_id": "nunique",
        })
    )
    out = out.rename(columns={"installation_id": "n_installations"})

    # Filter valid rows (both verified and allocated must be positive)
    n_before = len(out)
    out = out.dropna(subset=["eu_verified_tco2", "eu_alloc_total_tco2"], how="all")
    out = out[(out["eu_verified_tco2"] > 0) & (out["eu_alloc_total_tco2"] > 0)]
    
    # Derived metrics
    out["eu_shortfall_tco2"] = out["eu_verified_tco2"] - out["eu_alloc_total_tco2"]
    out["eu_alloc_ratio"] = out["eu_alloc_total_tco2"] / out["eu_verified_tco2"].replace(0, np.nan)

    # Filter year range first
    out = out[(out["year"] >= start_year) & (out["year"] <= end_year)]
    
    # Filter outlier allocation ratios (keep within specified range)
    n_before_ratio = len(out)
    out = out[(out["eu_alloc_ratio"] >= alloc_ratio_min) & (out["eu_alloc_ratio"] <= alloc_ratio_max)]
    n_ratio_filtered = n_before_ratio - len(out)
    if n_ratio_filtered > 0:
        print(f"Allocation ratio filter ({alloc_ratio_min}x–{alloc_ratio_max}x): dropped {n_ratio_filtered:,} outlier rows")
    
    # Facility-level emissions filter: keep facilities with at least one year above threshold
    facs_above_threshold = set(out[out["eu_verified_tco2"] >= min_verified_co2_tco2][fac_id_col].unique())
    n_facs_before = out[fac_id_col].nunique()
    out = out[out[fac_id_col].isin(facs_above_threshold)]
    n_facs_after = out[fac_id_col].nunique()
    
    n_after = len(out)
    print(f"Emissions filter (≥{min_verified_co2_tco2/1000:.0f} kt in any year): {n_facs_before:,} → {n_facs_after:,} facilities")
    print(f"Facility-years: {n_before:,} → {n_after:,} rows")

    keep_cols = [fac_id_col, "year", "n_installations",
                 "eu_verified_tco2", "eu_alloc_total_tco2",
                 "eu_surrendered_tco2", "eu_shortfall_tco2", "eu_alloc_ratio"]
    return out[keep_cols].sort_values([fac_id_col, "year"]), matched_ets_ids_per_fac
