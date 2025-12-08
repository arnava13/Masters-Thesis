"""
AlphaEarth satellite image embeddings via Google Earth Engine.

Provides 64-dimensional embeddings from the GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
ImageCollection, sampled at 10km scale around facility locations. These embeddings
encode land use, built environment, and natural features from satellite imagery.

Reference:
- Google Earth Engine Catalog: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
"""

from __future__ import annotations

import os
import ee
import pandas as pd
from pandas import DataFrame


# =============================================================================
# CONFIGURATION
# =============================================================================

ALPHA_EARTH_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
EMBEDDING_SCALE_M = 10_000  # 10km - EE averages the cell automatically
N_EMBEDDING_DIMS = 64


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _ensure_ee_initialized() -> None:
    """Ensure Earth Engine is initialized."""
    try:
        ee.Number(1).getInfo()  # type: ignore
    except Exception:
        try:
            ee.Initialize()
        except Exception:
            ee.Authenticate()
            ee.Initialize()


def _get_annual_mosaic(year: int, pt):
    """Get mosaicked embedding image for a year, filtered to point location."""
    return (
        ee.ImageCollection(ALPHA_EARTH_COLLECTION)  # type: ignore
        .filterDate(f"{year}-01-01", f"{year+1}-01-01")
        .filterBounds(pt)
        .mosaic()
    )


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def sample_embeddings_for_year(
    df_fac: DataFrame,
    fac_ids: set,
    year: int,
    fac_id_col: str = "idx",
    scale: int = EMBEDDING_SCALE_M,
) -> DataFrame:
    """
    Sample AlphaEarth embeddings at specified scale for given facilities.
    
    Args:
        df_fac: Static facilities DataFrame with lat, lon, fac_id_col columns
        fac_ids: Set of facility IDs to sample
        year: Year to sample embeddings for
        fac_id_col: Name of facility ID column
        scale: Sampling scale in meters (default: 10km)
        
    Returns:
        DataFrame with fac_id_col, year, and emb_00 through emb_63 columns
    """
    _ensure_ee_initialized()
    
    facs_to_sample = df_fac[df_fac[fac_id_col].isin(fac_ids)] # type: ignore
    
    if facs_to_sample.empty:
        return DataFrame()
    
    rows = []
    
    for i, (_, fac) in enumerate(facs_to_sample.iterrows()):
        fac_id = int(fac[fac_id_col])
        lon, lat = float(fac["lon"]), float(fac["lat"])
        
        pt = ee.Geometry.Point([lon, lat])  # type: ignore
        img = _get_annual_mosaic(year, pt)
        
        vals = img.reduceRegion(
            reducer=ee.Reducer.mean(),  # type: ignore
            geometry=pt,
            scale=scale
        ).getInfo()
        
        if vals and any(v is not None for v in vals.values()):
            vals[fac_id_col] = fac_id
            vals["year"] = year
            rows.append(vals)
        elif i == 0:
            print(f"    [{year}] First facility {fac_id} at ({lat:.2f}, {lon:.2f}): no data")
        
        if (i + 1) % 50 == 0:
            print(f"    [{year}] Processed {i+1}/{len(facs_to_sample)} facilities, {len(rows)} valid...")
    
    if not rows:
        return DataFrame()
    
    out = DataFrame(rows)
    rename_map = {f"A{i:02d}": f"emb_{i:02d}" for i in range(N_EMBEDDING_DIMS)}
    out = out.rename(columns=rename_map)
    
    return out


def build_embedding_panel(
    facilities_static: DataFrame,
    facilities_yearly: DataFrame,
    fac_id_col: str = "idx",
    cache_dir: str = "out",
    scale: int = EMBEDDING_SCALE_M,
) -> DataFrame:
    """
    Build AlphaEarth embedding panel for all facility-years with caching.
    
    Downloads embeddings year-by-year with incremental caching to handle GEE
    rate limits and allow resumption after failures.
    
    Args:
        facilities_static: Static facilities DataFrame with lat, lon, fac_id_col
        facilities_yearly: Yearly panel DataFrame with fac_id_col, year columns
        fac_id_col: Name of facility ID column
        cache_dir: Directory for cache files (default: 'out')
        scale: Sampling scale in meters (default: 10km)
        
    Returns:
        DataFrame with fac_id_col, year, and emb_00 through emb_63 columns
    """
    _ensure_ee_initialized()
    
    os.makedirs(cache_dir, exist_ok=True)
    
    final_fac_ids = set(facilities_yearly[fac_id_col].unique())
    facs = facilities_static[facilities_static[fac_id_col].isin(final_fac_ids)].copy() # type: ignore
    
    fac_years_by_year = facilities_yearly.groupby("year")[fac_id_col].apply(set).to_dict()
    
    print(f"AlphaEarth: Sampling at {scale/1000:.0f}km scale ({N_EMBEDDING_DIMS} dimensions)")
    print(f"Downloading embeddings for {len(facilities_yearly):,} facility-years")
    print(f"Years: {sorted(fac_years_by_year.keys())}")
    
    all_emb_dfs = []
    
    for year in sorted(fac_years_by_year.keys()):
        year = int(year)
        fac_ids_for_year = fac_years_by_year[year]
        cache_path = os.path.join(cache_dir, f"emb_{year}.parquet")
        
        cached_df = None
        cached_facs: set = set()
        
        if os.path.exists(cache_path):
            try:
                cached_df = pd.read_parquet(cache_path)
                if "emb_00" in cached_df.columns and cached_df["emb_00"].notna().any(): # type: ignore
                    cached_facs = set(cached_df[cached_df["emb_00"].notna()][fac_id_col].unique()) # type: ignore
                else:
                    print(f"[{year}] Cache exists but has no valid embeddings, re-downloading...")
                    cached_df = None
            except Exception as e:
                print(f"[{year}] Error reading cache: {e}")
                cached_df = None
        
        missing_facs = fac_ids_for_year - cached_facs
        
        if not missing_facs:
            print(f"[{year}] Using cached data ({len(cached_df):,} rows)") # type: ignore
            all_emb_dfs.append(cached_df[cached_df[fac_id_col].isin(fac_ids_for_year)]) # type: ignore
            continue
        
        if cached_facs:
            print(f"[{year}] Cache has {len(cached_facs):,} facilities, downloading {len(missing_facs):,} missing...")
        else:
            print(f"[{year}] No cache, downloading {len(missing_facs):,} facilities...")
        
        df_new = sample_embeddings_for_year(facs, missing_facs, year, fac_id_col, scale) # type: ignore
        
        if df_new.empty and not cached_facs:
            print(f"[{year}] WARNING: No data downloaded and no cache!")
            continue
        
        if cached_df is not None and not cached_df.empty:
            cached_keep = cached_df[
                (cached_df[fac_id_col].isin(fac_ids_for_year)) & (cached_df["emb_00"].notna())
            ]
            if not df_new.empty:
                df_year = pd.concat([cached_keep, df_new], ignore_index=True)
            else:
                df_year = cached_keep
        else:
            df_year = df_new
        
        if not df_year.empty:
            df_year.to_parquet(cache_path, index=False)
            all_emb_dfs.append(df_year)
            n_valid = df_year["emb_00"].notna().sum() if "emb_00" in df_year.columns else 0 # type: ignore
            print(f"[{year}] Saved {len(df_year):,} rows ({n_valid} valid) to {cache_path}")
        else:
            print(f"[{year}] No data to save!")
    
    if not all_emb_dfs:
        print("No embedding data available!")
        return DataFrame()
    
    emb_panel = pd.concat(all_emb_dfs, ignore_index=True)
    emb_cols = [c for c in emb_panel.columns if c.startswith("emb_")]
    n_valid = emb_panel["emb_00"].notna().sum() if "emb_00" in emb_panel.columns else 0
    
    print(f"\nCombined embedding panel: {len(emb_panel):,} rows, {emb_panel[fac_id_col].nunique():,} facilities, {len(emb_cols)} dimensions")
    print(f"  Valid embeddings: {n_valid:,} ({100*n_valid/len(emb_panel):.1f}%)")
    
    return emb_panel


def test_embedding_sample(
    facilities_static: DataFrame,
    fac_id_col: str = "idx",
    test_year: int = 2022,
    scale: int = EMBEDDING_SCALE_M,
) -> dict | None:
    """
    Test AlphaEarth embedding download with a single European facility.
    
    Args:
        facilities_static: Static facilities DataFrame
        fac_id_col: Name of facility ID column
        test_year: Year to test (default: 2022)
        scale: Sampling scale in meters
        
    Returns:
        Dictionary with test values, or None if test failed
    """
    _ensure_ee_initialized()
    
    euro_facs = facilities_static[
        (facilities_static["lat"] > 40) & (facilities_static["lat"] < 60) &
        (facilities_static["lon"] > -10) & (facilities_static["lon"] < 30)
    ]
    test_fac = euro_facs.iloc[0] if len(euro_facs) > 0 else facilities_static.iloc[0]
    
    print(f"Testing facility {test_fac[fac_id_col]} at ({test_fac['lat']:.2f}, {test_fac['lon']:.2f})")
    
    test_pt = ee.Geometry.Point([float(test_fac["lon"]), float(test_fac["lat"])])  # type: ignore
    test_img = _get_annual_mosaic(test_year, test_pt)
    
    test_vals = test_img.reduceRegion(
        reducer=ee.Reducer.mean(),  # type: ignore
        geometry=test_pt,
        scale=scale
    ).getInfo()
    
    if test_vals and test_vals.get("A00") is not None:
        print(f"Test result: A00={test_vals['A00']:.4f}, A01={test_vals.get('A01'):.4f}")
        return test_vals
    else:
        print("Test failed!")
        return None
