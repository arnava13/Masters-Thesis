"""
clusters.py
===========
Assigns facilities to PyPSA-Eur power system regions.

PyPSA-Eur uses k-means clustering on the European transmission network
to create power system regions. These regions are used for:
- Clustered standard errors (account for within-region correlation)
- Region fixed effects (absorb regional shocks)
- Region × Year fixed effects (for TWFE specification 2)

Reference: https://pypsa-eur.readthedocs.io/
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from typing import Optional, Dict
import warnings

from .config import (
    FAC_ID_COL, PYPSA_CLUSTERS_DIR, CLUSTER_COL,
    LAT_COL, LON_COL, PYPSA_RESOLUTIONS
)


def load_pypsa_regions(
    clusters_dir: Optional[Path] = None,
    resolution: str = "128"
) -> gpd.GeoDataFrame:
    """
    Load PyPSA-Eur k-means clustered power system regions.
    
    Parameters
    ----------
    clusters_dir : Path, optional
        Directory containing cluster geojsons. Default: data/pypsa_clusters/
    resolution : str
        Cluster resolution (37, 64, 128, or 256 nodes for Europe)
    
    Returns
    -------
    GeoDataFrame with cluster geometries
    
    Notes
    -----
    Expected file: regions_onshore_elec_s{resolution}.geojson
    Download from PyPSA-Eur: https://github.com/PyPSA/pypsa-eur
    """
    clusters_dir = clusters_dir or PYPSA_CLUSTERS_DIR
    
    if resolution not in PYPSA_RESOLUTIONS:
        warnings.warn(f"Resolution {resolution} not in {PYPSA_RESOLUTIONS}")
    
    # Try different filename patterns
    patterns = [
        f"regions_onshore_elec_s{resolution}.geojson",
        f"regions_onshore_s{resolution}.geojson",
        f"clusters_{resolution}.geojson",
    ]
    
    for pattern in patterns:
        filepath = Path(clusters_dir) / pattern
        if filepath.exists():
            gdf = gpd.read_file(filepath)
            print(f"Loaded {len(gdf)} PyPSA regions from {filepath.name}")
            return gdf
    
    raise FileNotFoundError(
        f"No cluster file found in {clusters_dir}. "
        f"Expected one of: {patterns}. "
        f"Download from https://github.com/PyPSA/pypsa-eur"
    )


def assign_pypsa_clusters(
    panel: pd.DataFrame,
    static: pd.DataFrame,
    resolution: str = "128",
    clusters_dir: Optional[Path] = None,
    cluster_col: str = CLUSTER_COL,
    id_col: str = FAC_ID_COL,
    lat_col: str = LAT_COL,
    lon_col: str = LON_COL
) -> pd.DataFrame:
    """
    Assign facilities to PyPSA-Eur power system regions via spatial join.
    
    Parameters
    ----------
    panel : DataFrame
        Panel data (facility × year)
    static : DataFrame
        Static facility data with lat/lon coordinates
    resolution : str
        PyPSA cluster resolution (37, 64, 128, 256)
    clusters_dir : Path, optional
        Directory with cluster geojsons
    cluster_col : str
        Output column name for cluster assignment
    id_col : str
        Facility ID column
    lat_col, lon_col : str
        Coordinate columns in static data
    
    Returns
    -------
    DataFrame
        Panel with cluster_col added
    """
    # Load regions
    regions = load_pypsa_regions(clusters_dir, resolution)
    
    # Create GeoDataFrame from static facility locations
    facilities = static[[id_col, lat_col, lon_col]].drop_duplicates(subset=[id_col])
    facilities_gdf = gpd.GeoDataFrame(
        facilities,
        geometry=gpd.points_from_xy(facilities[lon_col], facilities[lat_col]),
        crs="EPSG:4326"
    )
    
    # Ensure same CRS
    if regions.crs != facilities_gdf.crs:
        regions = regions.to_crs(facilities_gdf.crs)
    
    # Spatial join: which region contains each facility?
    joined = gpd.sjoin(facilities_gdf, regions, how="left", predicate="within")
    
    # Extract cluster ID (use 'name' or 'index_right' depending on geojson structure)
    if "name" in joined.columns:
        cluster_ids = joined.set_index(id_col)["name"]
    elif "index_right" in joined.columns:
        cluster_ids = joined.set_index(id_col)["index_right"]
    else:
        # Use first non-geometry column from regions
        region_cols = [c for c in regions.columns if c != "geometry"]
        cluster_ids = joined.set_index(id_col)[region_cols[0]]
    
    # Map to panel
    panel = panel.copy()
    panel[cluster_col] = panel[id_col].map(cluster_ids)
    
    # Report coverage
    n_assigned = panel[cluster_col].notna().sum()
    n_total = len(panel)
    n_facilities = panel[id_col].nunique()
    n_clusters = panel[cluster_col].nunique()
    print(f"Assigned {n_assigned}/{n_total} obs to {n_clusters} PyPSA regions")
    
    # Handle unassigned (facilities outside PyPSA regions)
    if panel[cluster_col].isna().any():
        n_missing = panel[cluster_col].isna().sum()
        warnings.warn(f"{n_missing} observations not assigned to any region")
        # Assign to nearest region or use country fallback
        panel[cluster_col] = panel[cluster_col].fillna("unassigned")
    
    return panel


def get_cluster_summary(
    panel: pd.DataFrame,
    cluster_col: str = CLUSTER_COL,
    id_col: str = FAC_ID_COL
) -> pd.DataFrame:
    """
    Summary statistics by cluster.
    
    Returns DataFrame with facility counts and observation counts per cluster.
    """
    summary = panel.groupby(cluster_col).agg(
        n_facilities=(id_col, "nunique"),
        n_observations=(id_col, "count")
    ).reset_index()
    
    summary = summary.sort_values("n_facilities", ascending=False)
    return summary
