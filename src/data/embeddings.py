"""
AlphaEarth satellite image embeddings via Google Earth Engine.

Provides 64-dimensional embeddings from the GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
ImageCollection, sampled at 10km scale around facility locations. These embeddings
encode land use, built environment, and natural features from satellite imagery.

Uses batch processing via ee.batch.Export.table.toDrive() for efficiency.

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
EMBEDDING_SCALE_M = 5_000  # 5km - EE averages the cell automatically
N_EMBEDDING_DIMS = 64
DEFAULT_DRIVE_FOLDER = 'ee_embedding_exports'


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


def _build_facilities_fc(
    facilities_static: DataFrame,
    fac_id_col: str = 'idx',
) -> ee.FeatureCollection:  # type: ignore
    """Build GEE FeatureCollection from facilities DataFrame."""
    features = []
    for _, row in facilities_static.iterrows():
        pt = ee.Geometry.Point([float(row['lon']), float(row['lat'])])  # type: ignore
        feat = ee.Feature(pt, {  # type: ignore
            fac_id_col: int(row[fac_id_col]),
            'lat': float(row['lat']),
            'lon': float(row['lon']),
        })
        features.append(feat)
    return ee.FeatureCollection(features)  # type: ignore


# =============================================================================
# BATCH EXPORT FUNCTIONS
# =============================================================================

def export_embeddings_year_batch(
    facilities_static: DataFrame,
    year: int,
    fac_id_col: str = 'idx',
    drive_folder: str = DEFAULT_DRIVE_FOLDER,
    file_prefix: str = 'emb',
    scale: int = EMBEDDING_SCALE_M,
    start_immediately: bool = True,
) -> ee.batch.Task:  # type: ignore
    """
    Submit a batch export task to sample AlphaEarth embeddings for all facilities.
    
    Uses server-side reduceRegions() to process all facilities in parallel,
    then exports results to Google Drive as CSV.
    
    Args:
        facilities_static: DataFrame with lat, lon, fac_id_col columns
        year: Year to sample embeddings for
        fac_id_col: Name of facility ID column
        drive_folder: Google Drive folder for exports
        file_prefix: Prefix for output filename
        scale: Sampling scale in meters (default: 10km)
        start_immediately: Whether to start task immediately
        
    Returns:
        ee.batch.Task object
    """
    _ensure_ee_initialized()
    
    # Build facilities FeatureCollection
    facilities_fc = _build_facilities_fc(facilities_static, fac_id_col)
    
    # Get annual embedding mosaic
    emb_img = (
        ee.ImageCollection(ALPHA_EARTH_COLLECTION)  # type: ignore
        .filterDate(f"{year}-01-01", f"{year+1}-01-01")
        .mosaic()
    )
    
    # Sample embeddings at each facility location
    # reduceRegions processes all facilities in parallel on GEE servers
    sampled = emb_img.reduceRegions(
        collection=facilities_fc,
        reducer=ee.Reducer.mean(),  # type: ignore
        scale=scale,
    )
    
    # Add year property to each feature
    sampled = sampled.map(lambda f: f.set('year', year))
    
    # Export to Drive
    file_name = f'{file_prefix}_{year}'
    task = ee.batch.Export.table.toDrive(  # type: ignore
        collection=sampled,
        description=file_name,
        folder=drive_folder,
        fileNamePrefix=file_name,
        fileFormat='CSV',
    )
    
    if start_immediately:
        task.start()
        print(f"Started task: {file_name}")
    
    return task


def export_embeddings_multi_year(
    facilities_static: DataFrame,
    years: list[int],
    fac_id_col: str = 'idx',
    drive_folder: str = DEFAULT_DRIVE_FOLDER,
    file_prefix: str = 'emb',
    scale: int = EMBEDDING_SCALE_M,
    start_immediately: bool = True,
) -> list:  # type: ignore
    """
    Submit batch export tasks for multiple years.
    
    Args:
        facilities_static: DataFrame with lat, lon, fac_id_col columns
        years: List of years to process
        fac_id_col: Name of facility ID column
        drive_folder: Google Drive folder for exports
        file_prefix: Prefix for output filename
        scale: Sampling scale in meters
        start_immediately: Whether to start tasks immediately
        
    Returns:
        List of ee.batch.Task objects
    """
    tasks = []
    for year in years:
        task = export_embeddings_year_batch(
            facilities_static=facilities_static,
            year=year,
            fac_id_col=fac_id_col,
            drive_folder=drive_folder,
            file_prefix=file_prefix,
            scale=scale,
            start_immediately=start_immediately,
        )
        tasks.append(task)
    
    print(f"\nSubmitted {len(tasks)} embedding export tasks for years {min(years)}-{max(years)}")
    return tasks


def load_embedding_results(
    years: list[int],
    drive_download_dir: str | None = None,
    file_prefix: str = 'emb',
    fac_id_col: str = 'idx',
) -> DataFrame:
    """
    Load and combine batch export results from Google Drive downloads.
    
    After running export_embeddings_multi_year(), download the CSV files from
    Google Drive to drive_download_dir, then call this function.
    
    Args:
        years: List of years to load
        drive_download_dir: Directory containing downloaded CSVs from Drive
                           (defaults to ~/Downloads)
        file_prefix: Prefix used in export
        fac_id_col: Name of facility ID column
        
    Returns:
        DataFrame with fac_id_col, year, and emb_00 through emb_63 columns
    """
    import glob
    
    if drive_download_dir is None:
        drive_download_dir = os.path.expanduser('~/Downloads')
    
    all_dfs = []
    
    for year in years:
        pattern = os.path.join(drive_download_dir, f'{file_prefix}_{year}.csv')
        files = glob.glob(pattern)
        
        if not files:
            print(f"  Warning: No file found for {year} ({pattern})")
            continue
        
        try:
            df = pd.read_csv(files[0])
            all_dfs.append(df)
            print(f"  Loaded {len(df):,} rows for {year}")
        except Exception as e:
            print(f"  Warning: Failed to load {files[0]}: {e}")
    
    if not all_dfs:
        raise ValueError(f"No valid data loaded for years {years}")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Rename A00-A63 to emb_00-emb_63
    rename_map = {f"A{i:02d}": f"emb_{i:02d}" for i in range(N_EMBEDDING_DIMS)}
    combined = combined.rename(columns=rename_map)
    
    # Keep only relevant columns
    emb_cols = [c for c in combined.columns if c.startswith("emb_")]
    keep_cols = [fac_id_col, 'year', 'lat', 'lon'] + emb_cols
    combined = combined[[c for c in keep_cols if c in combined.columns]]
    
    # Ensure proper types
    combined[fac_id_col] = combined[fac_id_col].astype(int)
    combined['year'] = combined['year'].astype(int)
    
    n_valid = combined["emb_00"].notna().sum() if "emb_00" in combined.columns else 0 # type: ignore
    print(f"\nCombined embedding panel: {len(combined):,} rows, "
          f"{combined[fac_id_col].nunique():,} facilities, {len(emb_cols)} dimensions") # type: ignore
    print(f"  Valid embeddings: {n_valid:,} ({100*n_valid/len(combined):.1f}%)")
    
    return combined # type: ignore
