"""
Data processing utilities for LCP/ETS pipeline.

Contains:
- fuel_group_map: Maps raw fuel types to standardized fuel groups
- normalize_pyeutl_id: Normalizes pyeutl installation IDs
- normalize_eureg_id: Normalizes EU Registry ETS identifiers
- build_eu_ets_yearly_from_zip: Builds yearly ETS metrics from zip file
- get_installation_activity_info: Extracts activity sector info from installations

EU ETS Activity Codes (from activity_type.csv in EUETS.INFO data):
- 1: Combustion installations (>20 MW thermal) - Phase 1-2
- 20: Combustion of fuels - Phase 3+
These indicate electricity/heat generators.
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


# EU ETS Activity codes for combustion installations
# Source: Directive 2003/87/EC Annex I (as amended)
#   https://eur-lex.europa.eu/eli/dir/2003/87/2024-03-01
# See also: EEA EU ETS Data Viewer Background Note, Table 6-1
#   https://www.eea.europa.eu/data-and-maps/data/european-union-emissions-trading-scheme-12/eu-ets-background-note
ELECTRICITY_ACTIVITY_CODES = {1, 20}
# Code 1 (Phases 1-2, 2005-2012): "Combustion installations with a rated thermal input exceeding 20 MW"
# Code 20 (Phase 3+, 2013-present): "Combustion of fuels" - covers same installations under revised Annex I
#
# NOTE: These codes identify combustion installations broadly (power plants, CHP, industrial boilers,
# district heating), not electricity generators specifically. However, for the LCP (Large Combustion
# Plant) dataset used in this study, the registry primarily covers electricity-generating facilities,
# so this approximation effectively identifies the electricity sector for PyPSA-Eur cluster assignment.


def build_eu_ets_yearly_from_zip(
    fn_zip: str,
    fac_static,
    fac_id_col: str = 'idx',
    start_year: int = 2018,
    end_year: int = 2023,
    alloc_ratio_min: float = 0.01,
    alloc_ratio_max: float = 20.0,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Build yearly EU ETS metrics aggregated at the facility level.
    Uses normalized ID matching between EU Registry ETS IDs and pyeutl installation IDs.
    
    Args:
        fn_zip: Path to EU ETS zip file
        fac_static: facilities_static DataFrame with fac_id_col and ets_id (list) columns
        fac_id_col: Name of facility ID column (default: 'idx')
        start_year: Start year for filtering (default: 2018)
        end_year: End year for filtering (default: 2023)
        alloc_ratio_min: Minimum allocation ratio to keep (default: 0.01)
        alloc_ratio_max: Maximum allocation ratio to keep (default: 20.0)
    
    Returns:
        Tuple of:
        - yearly_panel DataFrame
        - matched_ets_ids dict mapping facility idx -> list of matched NORMALIZED ETS IDs
        - facility_activity dict mapping facility idx -> dict with activity info:
            - 'activity_ids': set of activity codes for matched installations
            - 'is_electricity': True if any matched installation is combustion (activity_id in {1, 20})
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
        return pd.DataFrame(), {}, {}
    
    # Track which normalised ETS IDs actually matched (for updating facilities_static)
    matched_ets_ids_per_fac = (
        matched.groupby(fac_id_col)["ets_id_norm"]
        .apply(lambda x: list(x.unique()))
        .to_dict()
    )
    
    # --- Extract activity info for heterogeneity analysis ---
    # Get activity_id from installation data
    if "activity_id" in df_inst.columns:
        inst_activity = df_inst.set_index("id")["activity_id"].to_dict()
        matched["activity_id"] = matched["installation_id"].map(inst_activity)
        
        # Aggregate activity info per facility
        def agg_activity(g):
            activity_ids = set(g["activity_id"].dropna().astype(int))
            is_electricity = bool(activity_ids & ELECTRICITY_ACTIVITY_CODES)
            return pd.Series({
                "activity_ids": activity_ids,
                "is_electricity": is_electricity
            })
        
        facility_activity = (
            matched.groupby(fac_id_col)
            .apply(agg_activity, include_groups=False)
            .to_dict(orient="index")
        )
        
        n_electricity = sum(1 for v in facility_activity.values() if v["is_electricity"])
        print(f"Activity info: {n_electricity:,} facilities are electricity generators (activity_id in {ELECTRICITY_ACTIVITY_CODES})")
    else:
        print("Warning: activity_id column not found in installation data")
        facility_activity = {}
    
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
    
    n_after = len(out)
    print(f"Final ETS panel: {out[fac_id_col].nunique():,} facilities, {n_after:,} facility-years")

    keep_cols = [fac_id_col, "year", "n_installations",
                 "eu_verified_tco2", "eu_alloc_total_tco2",
                 "eu_surrendered_tco2", "eu_shortfall_tco2", "eu_alloc_ratio"]
    return out[keep_cols].sort_values([fac_id_col, "year"]), matched_ets_ids_per_fac, facility_activity
