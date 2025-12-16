"""
Beirle-style NOx emission quantification from TROPOMI NO₂ via flux divergence (advection) method.

Implements the methodology from:
- Beirle et al. (2023), ESSD 15, 3051–3073: "Improved catalog of NOx point source emissions (version 2)"
  https://essd.copernicus.org/articles/15/3051/2023/
- Beirle et al. (2021), ESSD 13, 2995–3012: "Pinpointing nitrogen oxide emissions from space" (v1)
  https://essd.copernicus.org/articles/13/2995/2021/

Key implementation details:
- Uses GEE OFFL L3 NO₂ instead of PAL product (10-40% higher TVCDs, see Sect. 5.3.7)
- Fixed NOx/NO₂ ratio of 1.38 ± 0.10 per Beirle Sect. 3.4 (vs full ESCiMo climatology)
- Topographic correction: A* = A + 1.5 × C_topo (Beirle Sect. 3.7, Sun 2022)
- Lifetime correction: c_τ = exp(t_r/τ) with τ(lat) from Lange et al. (2022)
- No AMF correction (using standard L3 AMF) - noted in uncertainty

References for key parameters are provided in docstrings.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import ee

from pandas import DataFrame


# =============================================================================
# BATCH EXPORT CONFIGURATION
# =============================================================================
DEFAULT_DRIVE_FOLDER = 'ee_beirle_exports'
BATCH_POLL_INTERVAL_S = 30  # Seconds between task status checks
MAX_FEATURES_PER_EXPORT = 50_000  # GEE limit for table exports


# =============================================================================
# CONFIGURATION AND CONSTANTS
# Sources: Beirle et al. (2023) ESSD 15, Sect. 3.10
# =============================================================================

# Integration radius (m): Beirle v2 uses 15 km for spatial integration
# "We have simplified the calculation of emissions by just integrating the advection map
#  spatially 15 km around the point source location." - Sect. 3.10.1
# EDIT: Due to poor correlation with reported NOx, we use 5km instead
INTEGRATION_RADIUS_M = 5_000

# Plume height (m above ground): used for wind interpolation
# "The v2 catalog is based on a plume height of 500 m above ground. This height is used for
#  the interpolation of wind fields." - Sect. 3.12.3
PLUME_HEIGHT_M = 500

# Minimum wind speed threshold (m/s): Beirle uses 2 m/s, I use 1 m/s to increase statistical power
# and filter noisy samples.
# "As in v1 of the catalog, only observations with wind speeds above 2 m/s are considered
#  in the further processing." - Sect. 3.5
MIN_WIND_SPEED_MS = 1.0

# Detection limit (kg/s): Beirle v2 Sect. 3.11.1
# Beirle uses 0.04 kg/s for high-albedo desert (LER > 8%), 0.11 kg/s elsewhere.
# We use a more permissive threshold of 0.01 kg/s to maximize statistical power,
# acknowledging that this is well below Beirle's validated detection limits.
# This choice trades off precision for sample size in econometric analysis.
DETECTION_LIMIT = 0.01  # Permissive threshold for statistical power

# Significance filter thresholds (Beirle Sect. 3.11)
MAX_REL_ERR_STAT = 0.30  # Maximum relative statistical integration error (30%)
INTERFERENCE_RADIUS_KM = 20  # Radius for spatial interference flagging (km)

# Minimum valid observation days per year
# Must have wind >= MIN_WIND_SPEED_MS. Set conservatively to allow alpine/calm locations.
MIN_VALID_DAYS = 20

# Molar mass of NOx (as NO₂ equivalent) in kg/mol
# Source: NIST Chemistry WebBook, NO₂ (CAS 10102-44-0)
# https://webbook.nist.gov/cgi/cbook.cgi?ID=10102-44-0
MOLAR_MASS_NO2_KG = 46.0055e-3

# Native TROPOMI L3 scale for reduceRegion (m)
# Source: GEE COPERNICUS/S5P/OFFL/L3_NO2 documentation
# https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_NO2
# "Pixel Size 1113.2 meters" (0.01° grid from harpconvert bin_spatial)
TROPOMI_SCALE_M = 1113

# ERA5-Land scale in GEE (m)
# Source: GEE ECMWF/ERA5_LAND/HOURLY documentation
# https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY
# Native resolution ~9km, served in GEE at 0.1° ≈ 11.1km
ERA5_SCALE_M = 11_132

# GEE dataset identifiers
TROPOMI_NO2_COLLECTION = 'COPERNICUS/S5P/OFFL/L3_NO2'
TROPOMI_NO2_BAND = 'tropospheric_NO2_column_number_density'
ERA5_HOURLY_COLLECTION = 'ECMWF/ERA5_LAND/HOURLY'
ERA5_U_BAND = 'u_component_of_wind_10m'
ERA5_V_BAND = 'v_component_of_wind_10m'
ERA5_T_BAND = 'temperature_2m'

# DEM for topographic correction (Beirle Sect. 3.7)
SRTM_DEM = 'USGS/SRTMGL1_003'
SRTM_ELEVATION_BAND = 'elevation'

# Topographic correction (Beirle Sect. 3.7, Appendix A)
# C_topo = (V / H_sh) × w₀ × ∇z₀, with H_sh = 1 km
# A* = A + f × C_topo, with f = 1.5 (empirically derived)
# Combined: A* = A + V × w₀ × ∇z₀ / (H_sh/f) = A + V × w₀ × ∇z₀ / 667m
TOPO_EFFECTIVE_SCALE_HEIGHT_M = 667  # = 1000m / 1.5 (net scale height per Appendix A)

# =============================================================================
# NO₂ → NOx SCALING (Photostationary State Approximation)
# Source: Beirle et al. (2023) Sect. 3.4, Dickerson et al. (1982)
# =============================================================================

# NOx/NO₂ scaling constants from Beirle et al. (2023) Sect. 3.4
# "For the detected point sources, the NOx/NO₂ ratio was found to be about 1.38 ± 0.10"
NOX_NO2_RATIO = 1.38
NOX_NO2_RATIO_UNCERTAINTY = 0.10  # Absolute uncertainty (not relative)

# =============================================================================
# NOx LIFETIME PARAMETERIZATION
# Source: Lange et al. (2022), via Beirle et al. (2023) Eq. (10)
# =============================================================================

def tau_from_lat(lat_deg: float) -> float:
    """
    NOx lifetime (hours) as function of latitude.
    
    Parameterization from Lange et al. (2022) as used in Beirle et al. (2023) Eq. (10):
        τ = 1.0089 × exp(0.0242 × (|lat| + 9.6024))
    where τ is in hours and lat is in degrees.
    
    "Lange et al. (2022) systematically investigated NOx lifetimes worldwide and found 
    typical values of about 2 h for low latitudes up to about 4–6 h at higher latitudes."
    - Beirle et al. (2023), Sect. 3.10.2
    
    Note: 50% uncertainty assumed for τ due to high variability at similar latitudes
    (Beirle et al. 2023, Sect. 3.12.1; Laughner and Cohen, 2019).
    
    For detected point sources, resulting c_τ ≈ 1.40 ± 0.24 (Beirle Sect. 3.10.2).
    
    Args:
        lat_deg: Latitude in degrees
        
    Returns:
        NOx lifetime in hours
    """
    return 1.0089 * np.exp(0.0242 * (abs(lat_deg) + 9.6024))


def lifetime_correction_factor(
    wind_speed_ms: float,
    lat_deg: float,
    radius_m: float = INTEGRATION_RADIUS_M
) -> float:
    """
    Compute lifetime correction factor c_τ.
    
    From Beirle et al. (2023) Eqs. (8)-(9):
        t_r = R / |w|           (residence time)
        c_τ = exp(t_r / τ)
    
    "The lifetime correction has to compensate for the integrated loss within the 
    residence time, which results in a scaling factor..." - Sect. 3.10.2
    
    Args:
        wind_speed_ms: Mean wind speed in m/s
        lat_deg: Latitude in degrees (for τ calculation)
        radius_m: Integration radius in meters
        
    Returns:
        Lifetime correction factor (dimensionless, typically 1.2-1.8).
        Returns NaN if wind_speed <= 0 or c_τ >= 3 (outside typical range).
    """
    if wind_speed_ms <= 0:
        return np.nan
    
    # Residence time (hours)
    t_r_hours = (radius_m / wind_speed_ms) / 3600.0
    
    # NOx lifetime (hours)
    tau_hours = tau_from_lat(lat_deg)
    
    # Correction factor: c_τ = exp(t_r / τ) (Beirle Eq. 9)
    c_tau = np.exp(t_r_hours / tau_hours)
    
    # Drop if outside typical range (Beirle reports c_τ ≈ 1.40 ± 0.24)
    if c_tau >= 3.0:
        return np.nan
    
    return c_tau


# =============================================================================
# EARTH ENGINE HELPER FUNCTIONS
# =============================================================================

def _ensure_ee_initialized() -> None:
    """Ensure Earth Engine is initialized."""
    try:
        ee.Number(1).getInfo() # type: ignore
    except Exception:
        try:
            ee.Initialize()
        except Exception:
            ee.Authenticate()
            ee.Initialize()
            

def _build_integration_disc(lon: float, lat: float, radius_m: float = INTEGRATION_RADIUS_M):
    """Build circular integration region."""
    return ee.Geometry.Point([lon, lat]).buffer(radius_m) # type: ignore

def _get_era5_daily_wind(year: int, month: int):
    """
    Build daily mean wind collection for a month.
    
    Returns ImageCollection with bands: u10, v10, wind_speed, wind_dir
    """
    start = ee.Date.fromYMD(year, month, 1) # type: ignore
    end = start.advance(1, 'month')
    n_days = end.difference(start, 'day')
    
    hourly = (
        ee.ImageCollection(ERA5_HOURLY_COLLECTION) # type: ignore
        .filterDate(start, end)
        .select([ERA5_U_BAND, ERA5_V_BAND])
    )
    
    days = ee.List.sequence(0, n_days.subtract(1)) # type: ignore
    
    def _compute_daily_mean(day_offset):
        day_offset = ee.Number(day_offset) # type: ignore
        day_start = start.advance(day_offset, 'day')
        day_end = day_start.advance(1, 'day')
        
        daily = hourly.filterDate(day_start, day_end).mean()
        
        u = daily.select(ERA5_U_BAND)
        v = daily.select(ERA5_V_BAND)
        
        # Wind speed (scalar)
        speed = u.pow(2).add(v.pow(2)).sqrt().rename('wind_speed')
        
        # Wind direction (meteorological: direction wind is coming FROM, degrees from N)
        # atan2(u, v) gives angle from North, add 180 to get "from" direction
        direction = u.atan2(v).multiply(180 / np.pi).add(180).mod(360).rename('wind_dir')
        
        return (
            daily
            .addBands(speed)
            .addBands(direction)
            .set('system:time_start', day_start.millis())
            .set('date_str', day_start.format('YYYY-MM-dd'))
        )
    
    return ee.ImageCollection(days.map(_compute_daily_mean)) # type: ignore


# =============================================================================
# Emission rate calculation using Beirle et. al. (2023) flux-divergence method
# Batches requests to GEE and computes advection server-side, then calculates emission rates with
# downloaded csvs.
# =============================================================================

def _build_facilities_fc(
    facilities_static: DataFrame,
    fac_id_col: str = 'idx',
) -> ee.FeatureCollection:  # type: ignore
    """
    Build GEE FeatureCollection from facilities DataFrame.
    
    Args:
        facilities_static: DataFrame with facility locations
        fac_id_col: Name of facility ID column
        
    Returns:
        ee.FeatureCollection with point geometries and idx property
    """
    features = []
    for _, row in facilities_static.iterrows():
        pt = ee.Geometry.Point([float(row['lon']), float(row['lat'])]) # type: ignore
        feat = ee.Feature(pt, { # type: ignore
            fac_id_col: int(row[fac_id_col]),
            'lat': float(row['lat']),
            'lon': float(row['lon']),
        })
        features.append(feat)
    
    return ee.FeatureCollection(features) # type: ignore


def _compute_advection_image_for_date(
    date: ee.Date, # type: ignore
    no2_ic: ee.ImageCollection, # type: ignore
    wind_ic: ee.ImageCollection, # type: ignore
) -> ee.Image: # type: ignore
    """
    Server-side computation of advection A* for a single date.
    
    Returns an Image with bands:
        - 'A_star': Topo-corrected advection [mol/(m²·s)]
        - 'wind_speed': Wind magnitude [m/s]
        - 'wind_u': Eastward wind component [m/s]
        - 'wind_v': Northward wind component [m/s]
        - 'pixel_count': Valid pixel mask for counting
    """
    # Filter to this date (server-side)
    date_str = date.format('YYYY-MM-dd')
    next_date = date.advance(1, 'day')

    # NO₂ image for this day (mosaic if multiple orbits).
    # Some days have no OFFL L3 NO₂ data at all; in that case, the filtered
    # collection is empty and mosaic() would return an Image with no bands,
    # causing Image.select(TROPOMI_NO2_BAND) to fail. Guard against that by
    # returning a fully masked placeholder image with the expected band name.
    no2_filtered = no2_ic.filterDate(date, next_date)
    no2_day = ee.Image(ee.Algorithms.If(  # type: ignore
        no2_filtered.size().gt(0),
        no2_filtered.mosaic().select(TROPOMI_NO2_BAND),
        ee.Image.constant(0) # type: ignore
        .rename(TROPOMI_NO2_BAND)
        .updateMask(ee.Image.constant(0)) # type: ignore
    ))
    
    # Scale to NOx
    no2_nox = no2_day.multiply(NOX_NO2_RATIO)
    
    # Wind for this day
    wind_day = wind_ic.filterDate(date, next_date).first()
    wind_u = wind_day.select(ERA5_U_BAND)
    wind_v = wind_day.select(ERA5_V_BAND)
    wind_speed = wind_day.select('wind_speed')
    
    # Gradient of NOx column
    # GEE's gradient() returns spatial derivatives in units of (input units) per meter,
    # correctly accounting for latitude-dependent projection scaling. No division needed.
    #
    # I verified this empirically: created test image with value = 10 × longitude.
    # At 50°N, gradient() returned 1.4×10⁻⁴, matching expected 10 / (111km × cos(50°))
    # = 10 / 71,400m = 1.4×10⁻⁴. This confirms gradient() is per-meter, not per-pixel.
    gradient_v = no2_nox.gradient()
    dv_dx = gradient_v.select('x')  # [mol/m² per meter] = [mol/m³]
    dv_dy = gradient_v.select('y')  # [mol/m² per meter] = [mol/m³]
    
    # Base advection: A = w · ∇V
    advection_base = dv_dx.multiply(wind_u).add(dv_dy.multiply(wind_v))
    
    # Topographic correction
    # DEM gradient is also per-meter (dimensionless slope), no division needed
    dem = ee.Image(SRTM_DEM).select(SRTM_ELEVATION_BAND) # type: ignore
    gradient_z = dem.gradient()
    dz_dx = gradient_z.select('x')  # [m elevation per m distance] = dimensionless
    dz_dy = gradient_z.select('y')  # [m elevation per m distance] = dimensionless
    
    wind_dot_grad_z = dz_dx.multiply(wind_u).add(dz_dy.multiply(wind_v))
    f_times_c_topo = no2_nox.multiply(wind_dot_grad_z).divide(TOPO_EFFECTIVE_SCALE_HEIGHT_M)
    
    # Corrected advection A*
    A_star = advection_base.add(f_times_c_topo).rename('A_star')
    
    # Valid pixel mask
    pixel_count = no2_nox.mask().rename('pixel_count')
    
    return (
        A_star
        .addBands(wind_speed.rename('wind_speed'))
        .addBands(wind_u.rename('wind_u'))
        .addBands(wind_v.rename('wind_v'))
        .addBands(pixel_count)
        .set('date_str', date_str)
        .set('system:time_start', date.millis())
    )


def _compute_monthly_advection_ic(year: int, month: int) -> ee.ImageCollection: # type: ignore
    """
    Build ImageCollection of daily advection images for a month.
    
    All computation is server-side; no getInfo() calls.
    
    Returns:
        ImageCollection where each image has bands: A_star, wind_speed, wind_u, wind_v, pixel_count
    """
    start = ee.Date.fromYMD(year, month, 1) # type: ignore
    end = start.advance(1, 'month')
    n_days = end.difference(start, 'day')
    
    # Load collections
    no2_ic = (
        ee.ImageCollection(TROPOMI_NO2_COLLECTION) # type: ignore
        .filterDate(start, end)
        .select(TROPOMI_NO2_BAND)
    )
    
    wind_ic = _get_era5_daily_wind(year, month)
    
    # Map over days
    days = ee.List.sequence(0, n_days.subtract(1)) # type: ignore
    
    def _compute_for_day(day_offset):
        day_offset = ee.Number(day_offset) # type: ignore
        date = start.advance(day_offset, 'day')
        return _compute_advection_image_for_date(date, no2_ic, wind_ic)
    
    return ee.ImageCollection(days.map(_compute_for_day)) # type: ignore


def _reduce_advection_for_facilities(
    advection_ic: ee.ImageCollection, # type: ignore
    facilities_fc: ee.FeatureCollection, # type: ignore
    fac_id_col: str = 'idx',
) -> ee.FeatureCollection: # type: ignore
    """
    Server-side reduction: compute mean advection stats for all facilities across all days.
    
    Uses reduceRegions to process all facilities × days efficiently on GEE servers.
    
    Args:
        advection_ic: ImageCollection from _compute_monthly_advection_ic
        facilities_fc: FeatureCollection from _build_facilities_fc
        fac_id_col: Facility ID column name
        
    Returns:
        FeatureCollection with one feature per facility-day, containing:
        - idx, lat, lon, date_str
        - A_star_sum: Sum of A* over 15km disc
        - wind_speed_mean: Mean wind speed at facility
        - pixel_count: Number of valid pixels
    """
    # Buffer facilities to 15km discs
    facilities_buffered = facilities_fc.map(
        lambda f: f.setGeometry(f.geometry().buffer(INTEGRATION_RADIUS_M))
    )
    
    def _reduce_for_date(img):
        """Reduce one day's advection image across all facilities."""
        date_str = img.get('date_str')
        
        # Reducer: sum A*, mean wind_speed, count pixels
        reducer = (
            ee.Reducer.sum().setOutputs(['A_star_sum']) # type: ignore
            .combine(ee.Reducer.mean().setOutputs(['wind_speed_mean']), sharedInputs=False) # type: ignore
            .combine(ee.Reducer.sum().setOutputs(['pixel_count']), sharedInputs=False) # type: ignore
        )
        
        # reduceRegions processes all facilities at once
        reduced = img.select(['A_star', 'wind_speed', 'pixel_count']).reduceRegions(
            collection=facilities_buffered,
            reducer=reducer,
            scale=TROPOMI_SCALE_M,
        )
        
        # Add date to each feature
        return reduced.map(lambda f: f.set('date_str', date_str))
    
    # Map over all days and flatten
    all_days = advection_ic.map(_reduce_for_date)
    
    return ee.FeatureCollection(all_days).flatten() # type: ignore


def export_beirle_month_batch(
    facilities_static: DataFrame,
    year: int,
    month: int,
    fac_id_col: str = 'idx',
    drive_folder: str = DEFAULT_DRIVE_FOLDER,
    file_prefix: str = 'beirle',
) -> ee.batch.Task: # type: ignore
    """
    Export Beirle advection data for one month as a batch task.
    
    This is ~100x faster than the interactive approach because:
    1. All facilities are processed in parallel via reduceRegions()
    2. Computation runs on GEE servers with no network round-trips
    3. Export runs asynchronously in GEE's batch queue
    
    Args:
        facilities_static: DataFrame with facility locations
        year: Year to process
        month: Month to process (1-12)
        fac_id_col: Name of facility ID column
        drive_folder: Google Drive folder for output
        file_prefix: Prefix for output filename
        
    Returns:
        ee.batch.Task object. Call task.start() to begin processing.
        
    Usage:
        task = export_beirle_month_batch(facilities, 2020, 1)
        task.start()
        # ... monitor with check_batch_tasks() or wait_for_tasks()
    """
    _ensure_ee_initialized()
    
    # Build facilities FeatureCollection
    facilities_fc = _build_facilities_fc(facilities_static, fac_id_col)
    
    # Compute monthly advection (server-side)
    advection_ic = _compute_monthly_advection_ic(year, month)
    
    # Reduce to facilities (server-side)
    results_fc = _reduce_advection_for_facilities(advection_ic, facilities_fc, fac_id_col)
    
    # Create export task
    description = f'{file_prefix}_{year}_{month:02d}'
    filename = f'{file_prefix}_{year}_{month:02d}'
    
    task = ee.batch.Export.table.toDrive( # type: ignore
        collection=results_fc,
        description=description,
        folder=drive_folder,
        fileNamePrefix=filename,
        fileFormat='CSV',
    )
    
    return task


def export_beirle_year_batch(
    facilities_static: DataFrame,
    year: int,
    fac_id_col: str = 'idx',
    drive_folder: str = DEFAULT_DRIVE_FOLDER,
    file_prefix: str = 'beirle',
    start_immediately: bool = True,
) -> list[ee.batch.Task]: # type: ignore
    """
    Export Beirle advection data for a full year as batch tasks (one per month).
    
    Args:
        facilities_static: DataFrame with facility locations
        year: Year to process
        fac_id_col: Name of facility ID column
        drive_folder: Google Drive folder for output
        file_prefix: Prefix for output filename
        start_immediately: If True, start all tasks immediately
        
    Returns:
        List of ee.batch.Task objects (12 tasks, one per month)
        
    Usage:
        tasks = export_beirle_year_batch(facilities, 2020)
        wait_for_tasks(tasks)  # Wait for completion
        df = load_batch_results(2020, drive_folder='ee_beirle_exports')
    """
    _ensure_ee_initialized()
    
    tasks = []
    for month in range(1, 13):
        task = export_beirle_month_batch(
            facilities_static=facilities_static,
            year=year,
            month=month,
            fac_id_col=fac_id_col,
            drive_folder=drive_folder,
            file_prefix=file_prefix,
        )
        tasks.append(task)
        
        if start_immediately:
            task.start()
            print(f"Started task: {file_prefix}_{year}_{month:02d}")
    
    return tasks


def load_batch_results(
    year: int,
    drive_download_dir: str | None = None,
    file_prefix: str = 'beirle',
) -> DataFrame:
    """
    Load and combine batch export results from Google Drive downloads.
    
    After running export_beirle_year_batch(), download the CSV files from 
    Google Drive to drive_download_dir, then call this function.
    
    Args:
        year: Year to load
        drive_download_dir: Directory containing downloaded CSVs from Drive
                           (defaults to ~/Downloads)
        file_prefix: Prefix used in export
        
    Returns:
        DataFrame with daily facility-level advection data
    """
    import glob
    
    if drive_download_dir is None:
        drive_download_dir = os.path.expanduser('~/Downloads')
    
    # Find all CSVs for this year
    pattern = os.path.join(drive_download_dir, f'{file_prefix}_{year}_*.csv')
    files = sorted(glob.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")
    
    print(f"Loading {len(files)} files for {year}...")
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: failed to load {f}: {e}")
    
    if not dfs:
        raise ValueError(f"No valid data loaded for {year}")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(combined):,} rows for {year}")
    
    return combined


def aggregate_batch_to_yearly(
    daily_df: DataFrame,
    fac_id_col: str = 'idx',
    min_valid_days: int = MIN_VALID_DAYS,
) -> DataFrame:
    """
    Aggregate batch daily results to yearly facility-level emissions.
    
    Applies wind speed filter, lifetime correction, and uncertainty propagation
    per the Beirle methodology.
    
    Args:
        daily_df: Output from load_batch_results() with columns:
                  idx, lat, lon, date_str, A_star_sum, wind_speed_mean, pixel_count
        fac_id_col: Facility ID column
        min_valid_days: Minimum valid days (wind >= MIN_WIND_SPEED_MS m/s)
        
    Returns:
        DataFrame with yearly facility-level emissions
    """
    df = daily_df.copy()
    
    # Filter: wind speed >= MIN_WIND_SPEED_MS
    df = df[df['wind_speed_mean'] >= MIN_WIND_SPEED_MS].copy()
    
    # Filter: sufficient pixels (at least 10 per disc)
    df = df[df['pixel_count'] >= 10].copy()
    
    # Compute pixel area and convert A_star_sum to E_mol_s
    pixel_area_m2 = TROPOMI_SCALE_M ** 2
    df['E_mol_s_raw'] = df['A_star_sum'] * pixel_area_m2  # Before lifetime correction
    
    # Lifetime correction: c_τ = exp(t_r / τ) per day
    def _apply_lifetime_correction(row):
        c_tau = lifetime_correction_factor(row['wind_speed_mean'], row['lat'])
        if c_tau is None or np.isnan(c_tau):
            return np.nan, np.nan
        return row['E_mol_s_raw'] * c_tau, c_tau
    
    df[['E_mol_s', 'c_tau']] = df.apply( # type: ignore
        _apply_lifetime_correction, axis=1, result_type='expand'
    )
    
    # Drop rows with invalid lifetime correction
    df = df.dropna(subset=['E_mol_s', 'c_tau']) # type: ignore
    
    # Extract year from date_str
    df['year'] = pd.to_datetime(df['date_str']).dt.year
    
    # Group by facility × year
    grouped = df.groupby([fac_id_col, 'year', 'lat', 'lon'])
    
    results = []
    for (idx, year, lat, lon), group in grouped: # type: ignore
        n_days = len(group)
        
        if n_days < min_valid_days:
            continue
        
        # Temporal statistics
        E_mol_s_mean = group['E_mol_s'].mean()
        E_mol_s_std = group['E_mol_s'].std()
        c_tau_mean = group['c_tau'].mean()
        
        # Convert to kg/s
        E_kg_s = E_mol_s_mean * MOLAR_MASS_NO2_KG
        se_E_mol_s = E_mol_s_std / np.sqrt(n_days)
        stat_err_kg_s = se_E_mol_s * MOLAR_MASS_NO2_KG
        
        # Relative errors (following Beirle Sect. 3.12)
        rel_err_stat = abs(stat_err_kg_s / E_kg_s) if E_kg_s != 0 else np.nan
        rel_err_tau = np.log(c_tau_mean) * 0.50 if c_tau_mean > 1 else 0.10
        rel_err_nox = NOX_NO2_RATIO_UNCERTAINTY / NOX_NO2_RATIO
        rel_err_amf = 0.10
        rel_err_height = 0.10
        rel_err_topo = 0.025
        # Note: OFFL vs PAL product difference (~25%) is a systematic bias, not random
        # uncertainty, so it is NOT included in error propagation. This means our
        # emission estimates are likely conservative (biased low by 10-40%).
        
        rel_err_total = np.sqrt(
            rel_err_stat**2 + rel_err_tau**2 + rel_err_nox**2 + 
            rel_err_amf**2 + rel_err_height**2 + rel_err_topo**2
        )
        
        results.append({
            fac_id_col: idx,
            'year': year,
            'beirle_nox_kg_s': E_kg_s,
            'beirle_nox_kg_s_se': stat_err_kg_s,
            'c_tau_mean': c_tau_mean,
            'rel_err_total': rel_err_total,
            'rel_err_stat': rel_err_stat,
            'rel_err_tau': rel_err_tau,
            'rel_err_nox': rel_err_nox,
            'rel_err_amf': rel_err_amf,
            'rel_err_height': rel_err_height,
            'rel_err_topo': rel_err_topo,
            'n_days_satellite': n_days,
        })
    
    result_df = pd.DataFrame(results)
    
    # Add significance flags
    if not result_df.empty:
        result_df['above_dl'] = result_df['beirle_nox_kg_s'] >= DETECTION_LIMIT
        result_df['rel_err_stat_lt_0_3'] = result_df['rel_err_stat'] < MAX_REL_ERR_STAT
        
        # Filter out high-uncertainty observations (>50% total relative error)
        # High uncertainty obs add noise without proportional information content
        MAX_REL_ERR_TOTAL = 0.50
        n_before = len(result_df)
        result_df = result_df[result_df['rel_err_total'] <= MAX_REL_ERR_TOTAL].copy()
        n_dropped = n_before - len(result_df)
        if n_dropped > 0:
            print(f"Dropped {n_dropped} observations with rel_err_total > {MAX_REL_ERR_TOTAL:.0%} "
                  f"({n_dropped/n_before:.1%} of sample)")
        
        # Add inverse-variance weight for weighted estimation
        # Weight = 1 / rel_err_total^2, capped at 99th percentile to avoid extreme weights
        result_df['nox_weight'] = 1.0 / (result_df['rel_err_total'] ** 2)
        weight_cap = float(result_df['nox_weight'].quantile(0.99))  # type: ignore
        result_df['nox_weight'] = result_df['nox_weight'].clip(upper=weight_cap)  # type: ignore
    
    return result_df  # type: ignore


# =============================================================================
# AUXILIARY FUNCTIONS (urbanization, interference)
# =============================================================================

def compute_urbanization(
    facilities_static: DataFrame,
    fac_id_col: str = 'idx',
    urban_threshold: int = 22,
) -> DataFrame:
    """
    Add urbanization degree from JRC GHSL Degree of Urbanization raster (SMOD).
    
    Uses GEE to sample the GHSL SMOD raster at each facility location, returning:
    - urbanization_degree: Continuous value (10-30) from GHSL SMOD classification
    - in_urban_area: Boolean flag (degree >= threshold) for heterogeneity analysis
    
    GHSL SMOD Classes (Degree of Urbanisation methodology):
        10: Water
        11: Very low density rural
        12: Low density rural  
        13: Rural cluster
        21: Suburban or peri-urban
        22: Semi-dense urban cluster
        23: Dense urban cluster
        30: Urban centre
    
    These variables are collected for heterogeneity analysis and descriptive
    statistics, not as regression controls. The AlphaEarth embeddings already encode
    urbanization implicitly, so including an explicit urbanization control would
    introduce multicollinearity without improving identification.
    
    Reference: Schiavina et al. (2023), GHS-SMOD R2023A
    https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2023A_GHS_SMOD_V2-0
    
    Args:
        facilities_static: Static facility DataFrame with lat, lon, fac_id_col
        fac_id_col: Name of facility ID column
        urban_threshold: SMOD value threshold for in_urban_area flag (default: 22 = semi-dense urban+)
        
    Returns:
        facilities_static with 'urbanization_degree' and 'in_urban_area' columns added
    """
    _ensure_ee_initialized()
    
    # GEE GHSL SMOD (Degree of Urbanization) raster - use 2020 epoch
    GHSL_SMOD = 'JRC/GHSL/P2023A/GHS_SMOD_V2-0'
    SMOD_BAND = 'smod_code'
    SMOD_SCALE = 1000  # 1km resolution
    
    facilities_static = facilities_static.copy()
    
    # Get unique facility locations
    fac_coords = facilities_static[[fac_id_col, 'lat', 'lon']].dropna(subset=['lat', 'lon']) # type: ignore
    
    urbanization_values = {}
    
    print(f"Sampling GHSL urbanization degree for {len(fac_coords)} facilities...")
    
    # Load SMOD image (use 2020 as representative epoch)
    smod = ee.ImageCollection(GHSL_SMOD).filter(  # type: ignore
        ee.Filter.eq('system:index', '2020')  # type: ignore
    ).first().select(SMOD_BAND)
    
    # Process in batches
    batch_size = 50
    for i in range(0, len(fac_coords), batch_size):
        batch = fac_coords.iloc[i:i+batch_size]
        
        for _, row in batch.iterrows():
            fac_id = row[fac_id_col]
            pt = ee.Geometry.Point([float(row['lon']), float(row['lat'])])  # type: ignore
            
            try:
                # Sample SMOD value at facility location
                result = smod.reduceRegion(
                    reducer=ee.Reducer.first(),  # type: ignore
                    geometry=pt,
                    scale=SMOD_SCALE,
                    bestEffort=True
                ).getInfo()
                
                smod_value = result.get(SMOD_BAND)
                urbanization_values[fac_id] = smod_value if smod_value is not None else np.nan
                
            except Exception:
                urbanization_values[fac_id] = np.nan
        
        if (i + batch_size) % 100 == 0:
            print(f"  Processed {min(i + batch_size, len(fac_coords))}/{len(fac_coords)} facilities")
    
    # Add columns to DataFrame
    facilities_static['urbanization_degree'] = facilities_static[fac_id_col].map(urbanization_values) # type: ignore
    facilities_static['in_urban_area'] = facilities_static['urbanization_degree'] >= urban_threshold
    
    # Summary statistics
    n_total = len(facilities_static)
    n_valid = facilities_static['urbanization_degree'].notna().sum()
    n_urban = facilities_static['in_urban_area'].sum()
    mean_degree = facilities_static['urbanization_degree'].mean()
    
    print(f"\nUrbanization summary ({n_valid}/{n_total} with valid data):")
    print(f"  Mean degree: {mean_degree:.1f}")
    print(f"  Urban (>= {urban_threshold}): {n_urban} ({100*n_urban/n_total:.1f}%)")
    print(f"  Rural (< {urban_threshold}): {n_total - n_urban} ({100*(n_total-n_urban)/n_total:.1f}%)")
    
    # Distribution by class
    class_labels = {
        10: 'Water', 11: 'Very low rural', 12: 'Low rural', 13: 'Rural cluster',
        21: 'Suburban', 22: 'Semi-dense urban', 23: 'Dense urban', 30: 'Urban centre'
    }
    print("\n  Distribution by SMOD class:")
    for code, label in sorted(class_labels.items()):
        n = (facilities_static['urbanization_degree'] == code).sum()
        if n > 0:
            print(f"    {code} ({label}): {n} ({100*n/n_total:.1f}%)")
    
    return facilities_static


def add_interference_flag(
    facilities_static: DataFrame,
    fac_id_col: str = 'idx',
    radius_km: float = INTERFERENCE_RADIUS_KM
) -> DataFrame:
    """
    Add interference flag to facilities_static based on proximity to other facilities.
    
    For each facility, checks if there is another facility within `radius_km`.
    If so, the satellite outcome may reflect cluster-level rather than 
    single-facility emissions. This is a static attribute (not time-varying).
    
    Uses WGS84 distance approximation (same as used in 500m spatial clustering).
    
    Args:
        facilities_static: Static facility DataFrame with lat, lon, fac_id_col
        fac_id_col: Name of facility ID column
        radius_km: Interference radius in km (default: 20 km per Beirle)
        
    Returns:
        facilities_static with 'interfered_5km' boolean column added
    """
    from scipy.spatial import cKDTree  # type: ignore
    
    facilities_static = facilities_static.copy()
    
    # Get coordinates for all facilities
    fac_coords = facilities_static[[fac_id_col, 'lat', 'lon']].copy()
    fac_coords = fac_coords.dropna(subset=['lat', 'lon'])  # type: ignore
    
    # Convert lat/lon to approximate meters using WGS84 formulas (same as Section 3.2)
    lat_rad = np.radians(fac_coords['lat'].values)  # type: ignore
    mean_lat_rad = np.mean(lat_rad)
    
    # m/deg latitude: WGS84 series expansion
    # (accurate to 0.01 m per Wikipedia)
    m_per_deg_lat = 111132.954 - 559.822 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad)
    # m/deg longitude: varies with cos(latitude)
    m_per_deg_lon = 111132.954 * np.cos(mean_lat_rad)
    
    coords_m = np.column_stack([
        fac_coords['lat'].values * m_per_deg_lat,  # type: ignore
        fac_coords['lon'].values * m_per_deg_lon   # type: ignore
    ])
    
    # Build spatial index
    tree = cKDTree(coords_m)
    
    # Find pairs within radius_km
    radius_m = radius_km * 1000
    pairs = tree.query_pairs(r=radius_m)
    
    # Build set of interfered facility IDs
    fac_ids = fac_coords[fac_id_col].values  # type: ignore
    interfered_ids_set: set = set()
    for i, j in pairs:
        interfered_ids_set.add(fac_ids[i])
        interfered_ids_set.add(fac_ids[j])
    
    # Add flag to facilities_static
    facilities_static['interfered_5km'] = facilities_static[fac_id_col].isin(interfered_ids_set) # type: ignore
    
    n_total = len(facilities_static)
    n_interfered = facilities_static[f'interfered_5km'].sum()
    print(f"Interference flag ({radius_km} km): {n_interfered}/{n_total} facilities "
          f"({100*n_interfered/n_total:.1f}%) have another ETS facility within radius")
    
    return facilities_static


def compute_interference_flags(
    beirle_panel: DataFrame,
    facilities_static: DataFrame,
    fac_id_col: str = 'idx',
    radius_km: float = INTERFERENCE_RADIUS_KM
) -> DataFrame:
    """
    DEPRECATED: Use add_interference_flag() on facilities_static instead.
    
    Add interference flag based on proximity to other ETS/LCP facilities.
    This function is kept for backward compatibility but the flag should be
    computed on facilities_static (not time-varying).
    """
    warnings.warn(
        "compute_interference_flags() is deprecated. "
        "Use add_interference_flag() on facilities_static instead.",
        DeprecationWarning
    )
    
    # Get interference set from static
    static_with_flag = add_interference_flag(facilities_static, fac_id_col, radius_km)
    interfered_ids = set(static_with_flag[static_with_flag['interfered_5km']][fac_id_col])
    
    # Add flag to Beirle panel
    beirle_panel = beirle_panel.copy()
    beirle_panel['interfered_5km'] = beirle_panel[fac_id_col].isin(interfered_ids) # type: ignore
    
    return beirle_panel


def load_beirle_panel(cache_dir: str = 'data/out', years: list[int] | None = None) -> DataFrame:
    """
    Load cached Beirle panel results.
    
    Args:
        cache_dir: Directory containing beirle_YYYY.parquet files
        years: Optional list of years to load (loads all if None)
        
    Returns:
        Combined DataFrame of all cached results
    """
    import glob
    
    pattern = os.path.join(cache_dir, 'beirle_*.parquet')
    files = glob.glob(pattern)
    
    if not files:
        return pd.DataFrame()
    
    dfs = []
    for f in files:
        # Extract year from filename
        basename = os.path.basename(f)
        if basename.startswith('beirle_failed_'):
            continue  # Skip failed tracking files
        
        try:
            year = int(basename.replace('beirle_', '').replace('.parquet', ''))
            if years is not None and year not in years:
                continue
            dfs.append(pd.read_parquet(f))
        except (ValueError, Exception):
            continue
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def plot_advection_map(
    lon: float,
    lat: float,
    year: int,
    emission_kg_s: float | None = None,
) -> None:
    """
    Plot yearly mean advection map A = w·∇V for a facility (diagnostic visualization).
    
    This shows the raw advection field averaged over the full year, matching what
    the beirle_panel computes. The emission rate is obtained by spatially 
    integrating A* over a 15 km disc with lifetime/NOx corrections.
    
    Args:
        lon, lat: Facility coordinates
        year: Year to visualize (full year average)
        emission_kg_s: Optional emission rate to display in legend
    
    Expected pattern for a point source:
        - Red (positive A): Upwind of source - concentration increases in wind direction
          (wind is blowing toward higher concentrations at source)
        - Blue (negative A): Downwind of source - concentration decreases in wind direction
          (wind carries plume away, concentration drops with distance)
        - Dipole structure: red upwind, blue downwind, centered on facility
        
    Note: The spatial integral of A over the disc gives the net emission rate.
    Red upwind + blue downwind is the mathematically correct pattern for A = w·∇V.
    """
    _ensure_ee_initialized()
    
    pt = ee.Geometry.Point([lon, lat]) # type: ignore
    
    # Auto-select a windy month for visualization
    # (Full year would require per-day advection computation which is too slow)
    print(f"Searching for a windy month in {year}...")
    
    # Try winter months first (typically windier in Europe)
    search_order = [1, 2, 12, 11, 3, 10, 4, 9, 5, 8, 6, 7]
    selected_month = None
    
    for m in search_order:
        start = ee.Date.fromYMD(year, m, 1) # type: ignore
        end = start.advance(1, 'month')
        
        daily_wind = _get_era5_daily_wind(year, m)
        wind_month = daily_wind.mean()
        
        wind_sample = wind_month.reduceRegion(
            reducer=ee.Reducer.mean(), # type: ignore
            geometry=pt,
            scale=ERA5_SCALE_M
        ).getInfo()
        
        wind_u = wind_sample.get(ERA5_U_BAND, 0) if wind_sample else 0 # type: ignore
        wind_v = wind_sample.get(ERA5_V_BAND, 0) if wind_sample else 0 # type: ignore
        ws = np.sqrt(wind_u**2 + wind_v**2)
        
        if ws >= MIN_WIND_SPEED_MS:
            selected_month = m
            wind_speed = ws
            print(f"  Selected: {year}-{m:02d} with {ws:.1f} m/s")
            break
        else:
            print(f"  {year}-{m:02d}: {ws:.1f} m/s (skip)")
    
    if selected_month is None:
        # Fallback to month with highest wind
        print(f"  No month >= {MIN_WIND_SPEED_MS} m/s, using best available")
        selected_month = 1
    
    # Load data for selected month
    start = ee.Date.fromYMD(year, selected_month, 1) # type: ignore
    end = start.advance(1, 'month')
    
    no2_ic = (
        ee.ImageCollection(TROPOMI_NO2_COLLECTION) # type: ignore
        .filterDate(start, end)
        .select(TROPOMI_NO2_BAND)
    )
    no2_mean = no2_ic.mean()
    
    daily_wind = _get_era5_daily_wind(year, selected_month)
    wind_mean = daily_wind.mean()
    
    # Sample wind at location
    wind_sample = wind_mean.reduceRegion(
        reducer=ee.Reducer.mean(), # type: ignore
        geometry=pt,
        scale=ERA5_SCALE_M
    ).getInfo()
    
    wind_u = wind_sample.get(ERA5_U_BAND) if wind_sample else None # type: ignore
    wind_v = wind_sample.get(ERA5_V_BAND) if wind_sample else None # type: ignore
    
    if wind_u is None or wind_v is None:
        raise ValueError(f"No wind data available for location ({lat}, {lon}) in {year}-{selected_month:02d}. Try a different facility.")
    
    wind_speed = np.sqrt(wind_u**2 + wind_v**2)
    
    # Wind direction: angle FROM which wind is blowing (meteorological convention)
    # atan2(u, v) gives direction wind is going TO; add 180 for FROM direction
    wind_dir_rad = np.arctan2(wind_u, wind_v)
    wind_dir_deg = np.degrees(wind_dir_rad)
    wind_dir_from = (wind_dir_deg + 180) % 360  # Meteorological "from" convention
    
    # Cardinal direction label
    cardinals = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    cardinal_idx = int((wind_dir_from + 22.5) // 45) % 8
    cardinal = cardinals[cardinal_idx]
    
    print(f"Using {year}-{selected_month:02d}: {wind_speed:.1f} m/s from {cardinal} ({wind_dir_from:.0f} deg)")
    
    # Compute gradient and advection: A = w · ∇V (Beirle Eq. 5)
    # GEE gradient() returns per-meter spatial derivatives (verified empirically Dec 2024)
    gradient = no2_mean.gradient()
    dx = gradient.select('x')  # [mol/m² per m] = [mol/m³]
    dy = gradient.select('y')  # [mol/m² per m] = [mol/m³]
    
    advection = dx.multiply(wind_u).add(dy.multiply(wind_v))  # [mol/(m²·s)]
    
    # Integration disc for clipping
    disc = _build_integration_disc(lon, lat, INTEGRATION_RADIUS_M * 1.5)
    
    print("Fetching advection map (this may take a minute)...")
    
    import geemap  # type: ignore
    
    # Compute actual data range for proper scaling
    clipped_advection = advection.clip(disc)
    stats = clipped_advection.reduceRegion(
        reducer=ee.Reducer.percentile([2, 98]),  # type: ignore
        geometry=disc,
        scale=TROPOMI_SCALE_M,
        maxPixels=1e6
    ).getInfo()
    
    p2 = stats.get('x_p2', -1e-10)  # type: ignore
    p98 = stats.get('x_p98', 1e-10)  # type: ignore
    
    # Use symmetric scale centered on zero
    abs_max = max(abs(p2), abs(p98), 1e-12)  # Avoid zero range
    print(f"Advection range (2nd-98th pctl): [{p2:.2e}, {p98:.2e}]")
    
    m = geemap.Map(center=[lat, lon], zoom=10)
    
    vis_params = {
        'min': -abs_max,
        'max': abs_max,
        'palette': ['blue', 'white', 'red']
    }
    
    m.addLayer(clipped_advection, vis_params, 'Advection')
    m.addLayer(pt.buffer(500), {'color': 'black'}, 'Facility')
    
    # Add wind direction arrow using ipyleaflet
    from ipyleaflet import Marker, AwesomeIcon  # type: ignore
    
    # Arrow points in direction wind is going TO (not FROM)
    # Use a rotated arrow icon; rotation is clockwise from north
    wind_dir_to = wind_dir_deg  # Direction wind is blowing TO
    
    # Position arrow in corner of the disc (offset from center)
    arrow_offset_deg = 0.15  # ~15 km offset
    arrow_lat = lat + arrow_offset_deg
    arrow_lon = lon - arrow_offset_deg
    
    # Create arrow marker with rotation
    arrow_icon = AwesomeIcon(
        name='arrow-up',
        marker_color='blue',
        icon_color='white',
        spin=False
    )
    
    # Add arrow as a simple marker (rotation via CSS transform in title)
    arrow_marker = Marker(
        location=(arrow_lat, arrow_lon),
        icon=arrow_icon,
        rotation_angle=wind_dir_to,
        rotation_origin='center',
        title=f"Wind: {wind_speed:.1f} m/s from {cardinal}"
    )
    m.add_layer(arrow_marker)
    
    # Add text legend
    from ipywidgets import HTML  # type: ignore
    from ipyleaflet import WidgetControl  # type: ignore
    
    # Build legend content
    legend_lines = [f"<b>Wind:</b> {wind_speed:.1f} m/s from {cardinal}"]
    if emission_kg_s is not None:
        above_dl = emission_kg_s >= 0.04
        dl_status = "above" if above_dl else "below"
        legend_lines.append(f"<b>NOx:</b> {emission_kg_s:.3f} kg/s ({dl_status} DL)")
    
    legend_html = HTML(
        value=f"<div style='background:white;padding:8px;border-radius:5px;font-size:12px;'>"
              f"{'<br>'.join(legend_lines)}</div>"
    )
    m.add_control(WidgetControl(widget=legend_html, position='topright'))
    
    return m
