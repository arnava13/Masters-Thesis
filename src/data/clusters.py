"""
clusters.py
===========
Spatial clustering for panel analysis: NUTS regions and PyPSA-Eur power system regions.

Two clustering systems are supported:

1. **NUTS Regions** (Primary for all facilities, included in FE model.):
   - Eurostat administrative regions (NUTS2 level)
   - Used for: clustered standard errors, Region x Year fixed effects
   - Download: https://ec.europa.eu/eurostat/web/gisco/geodata/reference-data/administrative-units-statistical-units/nuts

2. **PyPSA-Eur Regions** (Only used for electricity sector heterogeneity analysis.):
   - k-means clusters on European transmission network topology
   - Used for: electricity sector heterogeneity analysis
   - Reference: https://pypsa-eur.readthedocs.io/
"""

import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd

# Default values
FAC_ID_COL = "idx"
LAT_COL = "lat"
LON_COL = "lon"
CLUSTER_COL = "nuts2_region"
PYPSA_CLUSTER_COL = "pypsa_cluster"
NUTS_LEVEL = 2
PYPSA_RESOLUTION = "128"
_PYPSA_RESOLUTIONS = ["37", "64", "128", "256"]


# =============================================================================
# NUTS Regions (Primary clustering for all facilities)
# =============================================================================

def load_nuts_regions(
    nuts_path: Optional[str] = None,
    level: int = NUTS_LEVEL
) -> gpd.GeoDataFrame:
    """
    Load NUTS regions from Eurostat GeoJSON.
    
    Parameters
    ----------
    nuts_path : str, optional
        Path to NUTS GeoJSON file. Default: data/nuts/NUTS_RG_10M_2021_4326_LEVL_2.geojson
    level : int
        NUTS level (0=country, 1=major regions, 2=regional, 3=local). Default: 2
    
    Returns
    -------
    GeoDataFrame with NUTS region geometries
    
    Notes
    -----
    Download from Eurostat GISCO:
    https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_10M_2021_4326_LEVL_2.geojson
    """
    if nuts_path is None:
        raise ValueError(
            "nuts_path is required. Pass the path to NUTS GeoJSON file.\n"
            "Download from: https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/"
            f"NUTS_RG_10M_2021_4326_LEVL_{level}.geojson"
        )
    
    if not os.path.exists(nuts_path):
        raise FileNotFoundError(
            f"NUTS regions file not found: {nuts_path}\n"
            f"Download from: https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/"
            f"NUTS_RG_10M_2021_4326_LEVL_{level}.geojson\n"
            f"Place in: data/nuts/"
        )
    
    gdf = gpd.read_file(nuts_path)
    
    # Filter to requested level if LEVL_CODE column exists
    if "LEVL_CODE" in gdf.columns:
        gdf = gdf[gdf["LEVL_CODE"] == level]
    
    print(f"Loaded {len(gdf)} NUTS{level} regions from {os.path.basename(nuts_path)}")
    return gdf # type: ignore


def assign_nuts_regions(
    panel: pd.DataFrame,
    static: pd.DataFrame,
    nuts_path: Optional[str] = None,
    level: int = NUTS_LEVEL,
    cluster_col: str = CLUSTER_COL,
    id_col: str = FAC_ID_COL,
    lat_col: str = LAT_COL,
    lon_col: str = LON_COL
) -> pd.DataFrame:
    """
    Assign facilities to NUTS regions via spatial join.
    
    This is the PRIMARY clustering method used for:
    - Clustered standard errors (account for within-region correlation)
    - Region x Year fixed effects (absorb regional shocks)
    
    Parameters
    ----------
    panel : DataFrame
        Panel data (facility x year)
    static : DataFrame
        Static facility data with lat/lon coordinates
    nuts_path : Path, optional
        Path to NUTS GeoJSON file
    level : int
        NUTS level (default: 2 for regional)
    cluster_col : str
        Output column name for cluster assignment (default: 'nuts2_region')
    id_col : str
        Facility ID column
    lat_col, lon_col : str
        Coordinate columns in static data
    
    Returns
    -------
    DataFrame with cluster_col added
    """
    # Load NUTS regions
    regions = load_nuts_regions(nuts_path, level)
    
    # Create GeoDataFrame from static facility locations
    facilities = static[[id_col, lat_col, lon_col]].drop_duplicates(subset=[id_col]) # type: ignore
    facilities_gdf = gpd.GeoDataFrame(
        facilities,
        geometry=gpd.points_from_xy(facilities[lon_col], facilities[lat_col]),
        crs="EPSG:4326"
    )
    
    # Ensure same CRS
    if regions.crs != facilities_gdf.crs:
        regions = regions.to_crs(facilities_gdf.crs) # type: ignore
    
    # Spatial join: which NUTS region contains each facility?
    joined = gpd.sjoin(facilities_gdf, regions, how="left", predicate="within")
    
    # Extract NUTS ID (use NUTS_ID column)
    nuts_id_col = "NUTS_ID" if "NUTS_ID" in joined.columns else "id"
    cluster_ids = joined.set_index(id_col)[nuts_id_col]
    
    # Map to panel
    panel = panel.copy()
    panel[cluster_col] = panel[id_col].map(cluster_ids) # type: ignore
    
    # Report coverage
    n_assigned = panel[cluster_col].notna().sum()
    n_total = len(panel)
    n_facilities = panel[id_col].nunique()
    n_clusters = panel[cluster_col].nunique()
    print(f"Assigned {n_assigned}/{n_total} obs to {n_clusters} NUTS{level} regions")
    
    # Handle unassigned (facilities outside NUTS regions, e.g., non-EU)
    if panel[cluster_col].isna().any(): # type: ignore
        n_missing = panel[cluster_col].isna().sum()
        warnings.warn(f"{n_missing} observations not assigned to any NUTS region (likely non-EU)")
        # Use country_code as fallback
        if "country_code" in static.columns:
            country_map = static.set_index(id_col)["country_code"]
            panel[cluster_col] = panel[cluster_col].fillna(
                panel[id_col].map(country_map).apply(lambda x: f"NON_EU_{x}" if pd.notna(x) else "UNASSIGNED") # type: ignore
            )
        else:
            panel[cluster_col] = panel[cluster_col].fillna("UNASSIGNED")
    
    return panel


# =============================================================================
# PyPSA-Eur Regions (Heterogeneity analysis for electricity sector only)
# =============================================================================

def load_pypsa_regions(
    clusters_dir: Optional[str] = None,
    resolution: str = PYPSA_RESOLUTION,
) -> gpd.GeoDataFrame:
    """
    Load PyPSA-Eur k-means clustered power system regions.
    
    Parameters
    ----------
    clusters_dir : str, optional
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
    if clusters_dir is None:
        raise ValueError(
            "clusters_dir is required. Pass the path to PyPSA clusters directory.\n"
            "Expected file: regions_onshore_elec_s{resolution}.geojson"
        )
    
    if resolution not in _PYPSA_RESOLUTIONS:
        warnings.warn(f"Resolution {resolution} not in standard set {_PYPSA_RESOLUTIONS}")
    
    # Try different filename patterns
    patterns = [
        f"regions_onshore_elec_s{resolution}.geojson",
        f"regions_onshore_s{resolution}.geojson",
        f"clusters_{resolution}.geojson",
    ]
    
    for pattern in patterns:
        filepath = os.path.join(clusters_dir, pattern)
        if os.path.exists(filepath):
            gdf = gpd.read_file(filepath)
            print(f"Loaded {len(gdf)} PyPSA regions from {os.path.basename(filepath)}")
            return gdf
    
    raise FileNotFoundError(
        f"No cluster file found in {clusters_dir}. "
        f"Expected one of: {patterns}. "
        f"Download from https://github.com/PyPSA/pypsa-eur"
    )


def assign_pypsa_clusters(
    panel: pd.DataFrame,
    static: pd.DataFrame,
    resolution: str = PYPSA_RESOLUTION,
    clusters_dir: Optional[str] = None,
    cluster_col: str = PYPSA_CLUSTER_COL,
    id_col: str = FAC_ID_COL,
    lat_col: str = LAT_COL,
    lon_col: str = LON_COL
) -> pd.DataFrame:
    """
    Assign facilities to PyPSA-Eur power system regions via spatial join.
    
    NOTE: This is for ELECTRICITY SECTOR HETEROGENEITY ANALYSIS ONLY.
    For main clustering (SEs, FE), use assign_nuts_regions() instead.
    
    Parameters
    ----------
    panel : DataFrame
        Panel data (facility x year)
    static : DataFrame
        Static facility data with lat/lon coordinates
    resolution : str
        PyPSA cluster resolution (37, 64, 128, 256)
    clusters_dir : Path, optional
        Directory with cluster geojsons
    cluster_col : str
        Output column name for cluster assignment (default: 'pypsa_cluster')
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
    facilities = static[[id_col, lat_col, lon_col]].drop_duplicates(subset=[id_col]) # type: ignore
    facilities_gdf = gpd.GeoDataFrame(
        facilities,
        geometry=gpd.points_from_xy(facilities[lon_col], facilities[lat_col]),
        crs="EPSG:4326"
    )
    
    # Ensure same CRS
    if regions.crs != facilities_gdf.crs:
        regions = regions.to_crs(facilities_gdf.crs) # type: ignore
    
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
    panel[cluster_col] = panel[id_col].map(cluster_ids) # type: ignore
    
    # Report coverage
    n_assigned = panel[cluster_col].notna().sum()
    n_total = len(panel)
    n_facilities = panel[id_col].nunique()
    n_clusters = panel[cluster_col].nunique()
    print(f"Assigned {n_assigned}/{n_total} obs to {n_clusters} PyPSA regions")
    
    # Handle unassigned (facilities outside PyPSA regions)
    if panel[cluster_col].isna().any(): # type: ignore
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


# =============================================================================
# Combined Static Assignment (for Data Pipeline)
# =============================================================================

def assign_clusters_to_static(
    static_df: pd.DataFrame,
    nuts_path: Optional[str] = None,
    pypsa_dir: Optional[str] = None,
    pypsa_resolution: str = PYPSA_RESOLUTION,
    id_col: str = FAC_ID_COL,
    lat_col: str = LAT_COL,
    lon_col: str = LON_COL,
    nuts_col: str = CLUSTER_COL,
    pypsa_col: str = PYPSA_CLUSTER_COL
) -> pd.DataFrame:
    """
    Assign NUTS2 and PyPSA-Eur clusters to facilities_static via spatial join.
    
    This is the main entry point for the Data Pipeline notebook.
    
    Parameters
    ----------
    static_df : DataFrame
        Static facility data with lat/lon coordinates
    nuts_path : str, optional
        Path to NUTS GeoJSON file
    pypsa_dir : str, optional
        Directory containing PyPSA cluster geojsons
    pypsa_resolution : str
        PyPSA cluster resolution (default: "128")
    id_col, lat_col, lon_col : str
        Column names for facility ID and coordinates
    nuts_col : str
        Output column name for NUTS2 region (default: 'nuts2_region')
    pypsa_col : str
        Output column name for PyPSA cluster (default: 'pypsa_cluster')
    
    Returns
    -------
    DataFrame with nuts_col and pypsa_col added
    """
    static_df = static_df.copy()
    
    # Create GeoDataFrame from facilities
    facilities_gdf = gpd.GeoDataFrame(
        static_df,
        geometry=gpd.points_from_xy(static_df[lon_col], static_df[lat_col]),
        crs="EPSG:4326"
    )
    
    # --- NUTS2 Assignment (Primary) ---
    if nuts_path is not None and os.path.exists(nuts_path):
        nuts_gdf = gpd.read_file(nuts_path)
        if "LEVL_CODE" in nuts_gdf.columns:
            nuts_gdf = nuts_gdf[nuts_gdf["LEVL_CODE"] == NUTS_LEVEL]
        if nuts_gdf.crs != facilities_gdf.crs:
            nuts_gdf = nuts_gdf.to_crs(facilities_gdf.crs) # type: ignore
        
        joined = gpd.sjoin(facilities_gdf, nuts_gdf, how="left", predicate="within") # type: ignore
        nuts_id_col = "NUTS_ID" if "NUTS_ID" in joined.columns else "id"
        static_df[nuts_col] = joined[nuts_id_col].values
        
        n_missing = static_df[nuts_col].isna().sum()
        if n_missing > 0:
            print(f"NUTS2: {n_missing} facilities outside EU, using country fallback")
            static_df[nuts_col] = static_df[nuts_col].fillna(
                static_df["country_code"].apply(lambda x: f"NON_EU_{x}" if pd.notna(x) else "UNASSIGNED")
            )
        print(f"Assigned {len(static_df)} facilities to {static_df[nuts_col].nunique()} NUTS2 regions")
    else:
        if nuts_path is None:
            print("NUTS path not provided, using country_code as fallback")
        else:
            print(f"NUTS file not found: {nuts_path}")
        static_df[nuts_col] = static_df["country_code"]
    
    # --- PyPSA-Eur Assignment (Heterogeneity analysis) ---
    if pypsa_dir is not None:
        pypsa_file = os.path.join(pypsa_dir, f"regions_onshore_elec_s{pypsa_resolution}.geojson")
        if os.path.exists(pypsa_file):
            pypsa_gdf = gpd.read_file(pypsa_file)
            if pypsa_gdf.crs != facilities_gdf.crs:
                pypsa_gdf = pypsa_gdf.to_crs(facilities_gdf.crs) # type: ignore
            
            joined = gpd.sjoin(facilities_gdf, pypsa_gdf, how="left", predicate="within") # type: ignore
            pypsa_id_col = "name" if "name" in joined.columns else "index_right"
            static_df[pypsa_col] = joined[pypsa_id_col].values
            
            n_assigned = static_df[pypsa_col].notna().sum()
            n_clusters = static_df[pypsa_col].nunique()
            print(f"Assigned {n_assigned}/{len(static_df)} facilities to {n_clusters} PyPSA regions")
            
            if static_df[pypsa_col].isna().any(): # type: ignore
                static_df[pypsa_col] = static_df[pypsa_col].fillna("unassigned")
            # Convert to string for parquet compatibility (avoids mixed int/str types)
            static_df[pypsa_col] = static_df[pypsa_col].astype(str)
        else:
            print(f"PyPSA file not found: {pypsa_file} (skipping {pypsa_col})")
            static_df[pypsa_col] = None
    else:
        static_df[pypsa_col] = None
    
    return static_df
