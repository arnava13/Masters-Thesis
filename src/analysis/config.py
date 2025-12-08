"""
config.py
=========
Structural constants for the analysis module.

Contains paths and column names only. Analysis parameters (years, 
thresholds, outcomes) should be set in the notebook.

Outcomes:
- ETS CO₂: Verified annual CO₂ emissions from EU ETS registry (ground-truth).
- Satellite NOx: Beirle-style flux-divergence NOx proxy (noisy, with uncertainty).

Usage:
    from src.analysis.config import FAC_ID_COL, OUT_DIR, ...
"""

from pathlib import Path

# =============================================================================
# Directory Structure
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = DATA_DIR / "out"

# PyPSA-Eur k-means clustered power system regions
PYPSA_CLUSTERS_DIR = DATA_DIR / "pypsa_clusters"

# =============================================================================
# Data File Paths
# =============================================================================

FACILITIES_STATIC_PATH = OUT_DIR / "facilities_static.parquet"
FACILITIES_YEARLY_PATH = OUT_DIR / "facilities_yearly.parquet"
BEIRLE_PANEL_PATH = OUT_DIR / "beirle_panel.parquet"

# =============================================================================
# Column Names (structural - from Data Pipeline)
# =============================================================================

# Identifiers
FAC_ID_COL = "idx"
YEAR_COL = "year"

# Geographic
LAT_COL = "lat"
LON_COL = "lon"
COUNTRY_COL = "country_code"

# -----------------------------------------------------------------------------
# Treatment (ETS)
# -----------------------------------------------------------------------------
ALLOC_RATIO_COL = "eu_alloc_ratio"
ALLOC_COL = "eu_alloc_total_tco2"
SHORTFALL_COL = "eu_shortfall_tco2"

# -----------------------------------------------------------------------------
# Outcome 1: ETS Verified CO₂ (ground-truth administrative data)
# -----------------------------------------------------------------------------
ETS_CO2_COL = "eu_verified_tco2"          # Verified emissions (tCO2/yr)
LOG_ETS_CO2_COL = "log_ets_co2"           # Log-transformed (created in analysis)

# -----------------------------------------------------------------------------
# Outcome 2: Satellite NOx (Beirle-style flux-divergence proxy)
# Based on TROPOMI NO₂ + ERA5-Land winds. Noisy, with explicit uncertainties.
# -----------------------------------------------------------------------------
NOX_OUTCOME_COL = "beirle_nox_kg_s"       # NOx emission rate (kg/s)
NOX_SE_COL = "beirle_nox_kg_s_se"         # Standard error (kg/s)
NOX_REL_ERR_COL = "rel_err_total"         # Total relative uncertainty

# -----------------------------------------------------------------------------
# Satellite NOx Significance Flags (Beirle Sect. 3.11)
# We do NOT hard-filter; instead we use boolean flags for sensitivity analyses.
# This follows the principle that we skip Beirle's automatic point-source 
# identification (we have known ETS/LCP locations) but apply their quantification 
# and significance logic via these flags.
# -----------------------------------------------------------------------------

# Detection limit flags (Beirle Sect. 3.11.1)
DL_0_11_COL = "above_dl_0_11"       # >= 0.11 kg/s (standard, European conditions)
DL_0_03_COL = "above_dl_0_03"       # >= 0.03 kg/s (ideal: high-albedo desert)

# Statistical integration error flag (Beirle requires < 30%)
REL_ERR_STAT_LT_0_3_COL = "rel_err_stat_lt_0_3"

# Spatial interference flag: facility has another ETS facility within 20 km
# For these facilities, satellite outcome reflects cluster-level emissions
INTERFERED_20KM_COL = "interfered_20km"

# Legacy column names (for backward compatibility)
DL_CONSERVATIVE_COL = DL_0_11_COL
DL_PERMISSIVE_COL = DL_0_03_COL

# Individual uncertainty components (for diagnostics)
NOX_ERR_COMPONENTS = [
    "rel_err_stat", "rel_err_tau", "rel_err_nox", "rel_err_amf",
    "rel_err_height", "rel_err_topo", "rel_err_product"
]

# -----------------------------------------------------------------------------
# Plant characteristics
# -----------------------------------------------------------------------------
CAPACITY_COL = "capacity_mw"
FUEL_SHARE_COLS = [
    "share_gas", "share_coal", "share_oil",
    "share_biomass", "share_other_gas"
]

# Cluster column name (assigned by assign_pypsa_clusters)
CLUSTER_COL = "pypsa_cluster"

# =============================================================================
# PyPSA-Eur Resolutions (k-means cluster counts)
# =============================================================================

PYPSA_RESOLUTIONS = ["37", "64", "128", "256"]

# =============================================================================
# Callaway-Sant'Anna Defaults
# =============================================================================

CS_CONTROL_GROUP = "nevertreated"
CS_EST_METHOD = "dr"
CS_ANTICIPATION = 0
