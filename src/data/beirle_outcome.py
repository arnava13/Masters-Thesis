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
- Lifetime correction: c_τ = 1/(1 - exp(-t_r/τ)) with τ(lat) from Lange et al. (2022)
- No AMF correction (using standard L3 AMF) - noted in uncertainty

References for key parameters are provided in docstrings.
"""

from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd
import ee

from pandas import DataFrame


# =============================================================================
# CONFIGURATION AND CONSTANTS
# Sources: Beirle et al. (2023) ESSD 15, Sect. 3.10
# =============================================================================

# Integration radius (m): Beirle v2 uses 15 km for spatial integration
# "We have simplified the calculation of emissions by just integrating the advection map
#  spatially 15 km around the point source location." - Sect. 3.10.1
INTEGRATION_RADIUS_M = 15_000

# Plume height (m above ground): used for wind interpolation
# "The v2 catalog is based on a plume height of 500 m above ground. This height is used for
#  the interpolation of wind fields." - Sect. 3.12.3
PLUME_HEIGHT_M = 500

# Minimum wind speed threshold (m/s): Beirle uses 2 m/s
# "As in v1 of the catalog, only observations with wind speeds above 2 m/s are considered
#  in the further processing." - Sect. 3.5
MIN_WIND_SPEED_MS = 2.0

# Detection limits (kg/s): Beirle v2 Sect. 3.11.1
# Beirle uses 0.03 kg/s for high-albedo desert (LER > 8%), 0.11 kg/s elsewhere.
# We do not implement the LER mask, so we provide both thresholds for sensitivity.
# Europe is treated using the conservative 0.11 kg/s threshold for main regressions.
DETECTION_LIMIT_0_03 = 0.03   # Ideal conditions (high-albedo desert)
DETECTION_LIMIT_0_11 = 0.11   # Standard conditions (used for most European locations)

# Significance filter thresholds (Beirle Sect. 3.11)
MAX_REL_ERR_STAT = 0.30  # Maximum relative statistical integration error (30%)
INTERFERENCE_RADIUS_KM = 20  # Radius for spatial interference flagging (km)

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
# EMISSION RATE COMPUTATION (with lifetime correction)
# Source: Beirle et al. (2023) Sect. 3.5-3.7, Eq. 7, 8-9, 11
# =============================================================================

def _compute_emission_rate(
    no2_image,
    wind_u,
    wind_v,
    wind_speed: float,
    lat: float,
    disc,
    scale: float = TROPOMI_SCALE_M
) -> dict:
    """
    Compute lifetime-corrected NOx emission rate E (Beirle Eq. 11).
    
    Pipeline:
        1. Base advection: A = w · ∇V [mol/(m²·s)]
        2. Topographic correction: A* = A + f × C_topo [mol/(m²·s)]
        3. Spatial integration: E_raw = ∫∫ A* dx dy [mol/s]
        4. Lifetime correction: E = c_τ × E_raw [mol/s]
    
    Args:
        no2_image: NOx-scaled TVCD image (mol/m²), already multiplied by NOx/NO₂ ratio
        wind_u: Eastward wind component (m/s) as ee.Number
        wind_v: Northward wind component (m/s) as ee.Number
        wind_speed: Wind speed magnitude (m/s) for lifetime correction
        lat: Latitude (degrees) for lifetime parameterization
        disc: Integration geometry (15 km disc)
        scale: Pixel scale in meters
        
    Returns:
        Dictionary with:
        - 'E_mean': Mean A* × c_τ over disc [mol/(m²·s)] - for temporal averaging
        - 'E_count': Number of valid pixels
        - 'c_tau': Lifetime correction factor applied
    """
    try:
        # =====================================================================
        # 1. Base advection: A = u × (∂V/∂x) + v × (∂V/∂y)
        # =====================================================================
        gradient_v = no2_image.gradient()
        dv_dx = gradient_v.select('x').divide(scale)  # ∂V/∂x per meter
        dv_dy = gradient_v.select('y').divide(scale)  # ∂V/∂y per meter
        
        advection_base = (
            dv_dx.multiply(wind_u)
            .add(dv_dy.multiply(wind_v))
        )
        
        # =====================================================================
        # 2. Topographic correction (Beirle Sect. 3.7, Sun 2022)

        #    C_topo = (V / H_sh) · (w · ∇z₀)  where w·∇z₀ is dot product
        #    A* = A + f × C_topo, with f=1.5, H_sh=1km
        #    Combined: f × C_topo = V × (w · ∇z₀) / 667m
        # =====================================================================
        # Get DEM and compute elevation gradient ∇z₀
        dem = ee.Image(SRTM_DEM).select(SRTM_ELEVATION_BAND) # type: ignore
        gradient_z = dem.gradient()
        dz_dx = gradient_z.select('x').divide(30)  # SRTM is 30m resolution, ∂z₀/∂x
        dz_dy = gradient_z.select('y').divide(30)  # ∂z₀/∂y
        
        # Dot product: w · ∇z₀ = u × (∂z₀/∂x) + v × (∂z₀/∂y)
        wind_dot_grad_z = dz_dx.multiply(wind_u).add(dz_dy.multiply(wind_v))
        
        # f × C_topo = (f/H_sh) × V × (w · ∇z₀) = V × (w · ∇z₀) / 667m
        f_times_c_topo = (
            no2_image
            .multiply(wind_dot_grad_z)
            .divide(TOPO_EFFECTIVE_SCALE_HEIGHT_M)
        )
        
        # =====================================================================
        # 3. Corrected advection: A* = A + f × C_topo (Beirle Eq. 7)
        #    Units: A* has units mol/(m²·s) - advection flux per unit area
        # =====================================================================
        A_star = advection_base.add(f_times_c_topo)
        
        # =====================================================================
        # 4. Lifetime correction factor: c_τ (Beirle Eq. 8-9)
        #    Applied per-day using that day's wind speed
        # =====================================================================
        c_tau = lifetime_correction_factor(wind_speed, lat)
        
        # =====================================================================
        # 5. Lifetime-corrected advection: E_flux = c_τ × A* [mol/(m²·s)]
        # =====================================================================
        E_flux = A_star.multiply(c_tau).rename('E')
        
        # =====================================================================
        # 6. Spatial integration (Beirle Sect. 3.10.3, Eq. 11)
        #    E = c_τ × ∫∫ A* dx dy ≈ c_τ × Σ(A*_i) × pixel_area
        #    "summing up the advection values multiplied by the pixel area"
        # =====================================================================
        stats = E_flux.reduceRegion(
            reducer=ee.Reducer.sum().combine( # type: ignore
                ee.Reducer.count(), sharedInputs=True # type: ignore
            ),
            geometry=disc,
            scale=scale,
            bestEffort=True,
            maxPixels=1e6
        )
        
        result = stats.getInfo()
        
        # Convert sum to spatial integral: multiply by pixel area (scale²)
        pixel_area_m2 = scale * scale
        if result.get('E_sum') is not None:
            result['E_mol_s'] = result['E_sum'] * pixel_area_m2  # [mol/s]
        else:
            result['E_mol_s'] = None
        
        result['c_tau'] = c_tau  # Include c_tau in output for diagnostics
        return result
        
    except Exception as e:
        warnings.warn(f"Emission rate computation failed: {e}")
        return {'E_sum': None, 'E_count': None, 'E_mol_s': None, 'c_tau': None}


# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

def process_facility_day(
    lon: float,
    lat: float,
    date_str: str,
    no2_ic,
    daily_wind,
) -> dict | None:
    """
    Process a single facility-day to compute advection.
    
    Args:
        lon, lat: Facility coordinates
        date_str: Date string 'YYYY-MM-DD'
        no2_ic: Filtered NO₂ collection for the period (must include SZA band)
        daily_wind: Daily wind collection with wind_speed, wind_dir bands
        
    Returns:
        Dictionary with advection metrics, or None if no valid data
    """
    disc = _build_integration_disc(lon, lat)
    pt = ee.Geometry.Point([lon, lat]) # type: ignore
    
    # Get NO₂ and SZA for this day (mosaic multiple orbits)
    no2_day = no2_ic.filter(ee.Filter.eq('date_str', date_str)).mosaic() # type: ignore
    
    # Get wind for this day
    wind_day = daily_wind.filter(ee.Filter.eq('date_str', date_str)).first() # type: ignore
    
    if wind_day is None:
        return None
    
    # Sample wind at facility point
    wind_sample = wind_day.reduceRegion(
        reducer=ee.Reducer.mean(), # type: ignore
        geometry=pt,
        scale=ERA5_SCALE_M,
        bestEffort=True
    ).getInfo()
    
    wind_speed = wind_sample.get('wind_speed')
    if wind_speed is None or wind_speed < MIN_WIND_SPEED_MS:
        return None  # Calm day or no wind data
    
    wind_u = wind_sample.get(ERA5_U_BAND)
    wind_v = wind_sample.get(ERA5_V_BAND)
    
    if wind_u is None or wind_v is None:
        return None
    
    # Scale NO₂ to NOx using fixed ratio from Beirle et al. (2023) Sect. 3.4
    # "NOx/NO₂ ratio was found to be about 1.38 ± 0.10"
    no2_nox = no2_day.select(TROPOMI_NO2_BAND).multiply(NOX_NO2_RATIO)
    
    # Compute lifetime-corrected emission rate: E = c_τ × ∫∫ A* dx dy [mol/s]
    result = _compute_emission_rate(
        no2_nox,
        ee.Number(wind_u), # type: ignore
        ee.Number(wind_v), # type: ignore
        wind_speed,
        lat,
        disc
    )
    
    E_mol_s = result.get('E_mol_s')  # Spatial integral [mol/s]
    E_count = result.get('E_count')  # Number of valid pixels
    c_tau = result.get('c_tau')  # Lifetime correction factor
    
    if E_mol_s is None or E_count is None or E_count < 10:
        return None  # Insufficient data
    
    # Check for NaN c_tau (dropped by lifetime_correction_factor)
    if c_tau is None or np.isnan(c_tau):
        return None
    
    return {
        'date': date_str,
        'E_mol_s': E_mol_s,  # [mol/s] - spatially integrated emission rate
        'E_count': E_count,
        'c_tau': c_tau,
        'wind_speed_ms': wind_speed,
    }


def process_facility_year(
    idx: int,
    lon: float,
    lat: float,
    year: int,
    progress_callback=None
) -> dict | None:
    """
    Process a single facility-year to compute Beirle-style NOx emissions.
    
    Processes month-by-month to avoid GEE timeouts.
    
    Args:
        idx: Facility identifier
        lon, lat: Facility coordinates
        year: Year to process
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary with emission estimate and uncertainties, or None if insufficient data
    """
    _ensure_ee_initialized()
    
    # NOx/NO₂ ratio computed per-day from TROPOMI SZA (uncertainty from temporal SE)
    
    # Collect daily advection values across all months
    daily_results = []
    
    for month in range(1, 13):
        try:
            # Build collections for this month
            start = ee.Date.fromYMD(year, month, 1) # type: ignore
            end = start.advance(1, 'month')
            
            no2_ic = (
                ee.ImageCollection(TROPOMI_NO2_COLLECTION) # type: ignore
                .filterDate(start, end)
                .select(TROPOMI_NO2_BAND)
                .map(lambda img: img.set('date_str', img.date().format('YYYY-MM-dd')))
            )
            
            daily_wind = _get_era5_daily_wind(year, month)
            
            # Get list of dates with NO₂ data
            dates = no2_ic.aggregate_array('date_str').distinct().getInfo()
            
            for date_str in dates:
                try:
                    result = process_facility_day(
                        lon, lat, date_str, no2_ic, daily_wind
                    )
                    if result is not None:
                        daily_results.append(result)
                except Exception as e:
                    continue  # Skip failed days
                    
        except Exception as e:
            if progress_callback:
                progress_callback(f"Month {month} failed: {e}")
            continue
    
    if len(daily_results) < 30:  # Require at least 30 valid days
        return None
    
    # Aggregate daily results (E_mol_s already spatially integrated per-day)
    df = pd.DataFrame(daily_results)
    
    # Temporal mean of daily emission rates [mol/s]
    # Each daily E_mol_s = c_τ × ∫∫ A* dx dy (Beirle Eq. 11)
    E_mol_s_mean = df['E_mol_s'].mean()
    E_mol_s_std = df['E_mol_s'].std()
    n_days = len(df)
    
    # Mean lifetime correction factor (for uncertainty propagation)
    c_tau_mean = df['c_tau'].mean()
    
    # Statistical error of mean [mol/s]
    se_E_mol_s = E_mol_s_std / np.sqrt(n_days)
    
    # Final emission estimate: E [mol/s] → E [kg/s]
    E_kg_s = E_mol_s_mean * MOLAR_MASS_NO2_KG
    
    # =================
    # UNCERTAINTY COMPONENTS (following Beirle Sect. 3.12)
    # Note: We approximate statistical error using day-to-day variability of
    # integrated emissions, rather than Beirle's per-pixel SE propagation.
    # This is more conservative (captures meteorological variability too).
    # =================
    
    # 1. Statistical error of spatial integration (Sect. 3.12.2)
    #    Approximated via SE of temporal mean of daily E_mol_s values.
    #    Differs from Beirle's per-pixel SE propagation but is more conservative.
    stat_err_kg_s = se_E_mol_s * MOLAR_MASS_NO2_KG
    rel_err_stat = abs(stat_err_kg_s / E_kg_s) if E_kg_s != 0 else np.nan
    
    # 2. Lifetime correction uncertainty (Sect. 3.12.1)
    #    50% rel. uncertainty on τ, propagated through c_τ = exp(t_r/τ)
    #    Error propagation: rel_err_c_τ = ln(c_τ) × rel_err_τ
    #    Typically 10-20% for c_τ ≈ 1.4
    rel_err_tau = np.log(c_tau_mean) * 0.50 if c_tau_mean > 1 else 0.10
    
    # 3. NOx/NO₂ scaling uncertainty (Sect. 3.12.1)
    #    ±0.10 on 1.38 ratio, ~7.2%
    rel_err_nox = NOX_NO2_RATIO_UNCERTAINTY / NOX_NO2_RATIO
    
    # 4. AMF correction uncertainty (Sect. 3.12.1)
    #    UNMODELED STRUCTURAL UNCERTAINTY: We do not implement an explicit AMF
    #    correction. This 10% term is carried as a generic structural uncertainty
    #    following Beirle's error budget, representing potential bias not variance.
    rel_err_amf = 0.10
    
    # 5. Plume height uncertainty (Sect. 3.12.3)
    #    UNMODELED STRUCTURAL UNCERTAINTY: We do not implement plume-height-
    #    dependent wind interpolation. This 10% term represents the sensitivity
    #    to assumed plume height (500m vs 300m) as reported by Beirle.
    rel_err_height = 0.10
    
    # 6. Topographic correction uncertainty (Sect. 3.12.4)
    #    33% rel. uncertainty on f=1.5, typically <2.5% for flat terrain
    rel_err_topo = 0.025
    
    # 7. OFFL vs PAL product uncertainty (our addition, not in Beirle)
    #    OFFL provides 10-40% lower TVCDs than PAL; use 25% as proxy.
    #    This is a structural uncertainty representing potential systematic bias.
    rel_err_product = 0.25
    
    # Total relative uncertainty (quadrature sum, Sect. 3.12.5)
    # Beirle reports typical total of 20-40%; ours is ~35-45% with OFFL term
    rel_err_total = np.sqrt(
        rel_err_stat**2 + rel_err_tau**2 + rel_err_nox**2 + 
        rel_err_amf**2 + rel_err_height**2 + rel_err_topo**2 +
        rel_err_product**2
    )
    
    # Absolute standard error (statistical only)
    beirle_nox_kg_s_se = stat_err_kg_s
    
    return {
        'idx': idx,
        'year': year,
        'beirle_nox_kg_s': E_kg_s,
        'beirle_nox_kg_s_se': beirle_nox_kg_s_se,
        'rel_err_total': rel_err_total,
        # Individual error components for diagnostics
        'rel_err_stat': rel_err_stat,
        'rel_err_tau': rel_err_tau,
        'rel_err_nox': rel_err_nox,
        'rel_err_amf': rel_err_amf,
        'rel_err_height': rel_err_height,
        'rel_err_topo': rel_err_topo,
        'rel_err_product': rel_err_product,
        'n_days_satellite': n_days,
    }


def build_beirle_panel(
    facilities_static: DataFrame,
    years: list[int],
    fac_id_col: str = 'idx',
    cache_dir: str = 'data/out',
    resume: bool = True
) -> DataFrame:
    """
    Build Beirle-style NOx emission panel for all facilities across years.
    
    Implements incremental processing with caching to handle GEE rate limits
    and allow resumption after failures.
    
    Args:
        facilities_static: DataFrame with facility locations (must have lat, lon, fac_id_col)
        years: List of years to process
        fac_id_col: Name of facility ID column
        cache_dir: Directory for intermediate cache files
        resume: If True, resume from cached results
        
    Returns:
        DataFrame with facility × year panel of Beirle-style NOx emissions
    """
    from tqdm.auto import tqdm
    
    _ensure_ee_initialized()
    os.makedirs(cache_dir, exist_ok=True)
    
    all_results = []
    
    for year in years:
        cache_path = os.path.join(cache_dir, f'beirle_{year}.parquet')
        failed_path = os.path.join(cache_dir, f'beirle_failed_{year}.parquet')
        
        # Load existing results
        if resume and os.path.exists(cache_path):
            existing = pd.read_parquet(cache_path)
            done_ids = set(existing[fac_id_col].unique())
            all_results.append(existing)
        else:
            existing = pd.DataFrame()
            done_ids = set()
        
        # Load failed IDs to skip
        if resume and os.path.exists(failed_path):
            failed_df = pd.read_parquet(failed_path)
            failed_ids = set(failed_df[fac_id_col].unique())
        else:
            failed_ids = set()
        
        # Filter to remaining facilities
        remaining = facilities_static[
            ~facilities_static[fac_id_col].isin(done_ids | failed_ids) # type: ignore
        ]
        
        n_done = len(done_ids)
        n_failed = len(failed_ids)
        n_total = len(facilities_static)
        
        print(f"[{year}] {n_done} done, {n_failed} failed, {len(remaining)} remaining of {n_total}")
        
        if len(remaining) == 0:
            continue
        
        year_results = list(existing.to_dict('records')) if not existing.empty else []
        new_failed = list(failed_ids)
        
        for i, (_, row) in enumerate(tqdm(remaining.iterrows(), 
                                          total=n_total,
                                          initial=n_done + n_failed,
                                          desc=f"Year {year}")):
            fac_id = row[fac_id_col]
            
            try:
                result = process_facility_year(
                    idx=fac_id, # type: ignore
                    lon=float(row['lon']),
                    lat=float(row['lat']),
                    year=year
                )
                
                if result is not None:
                    year_results.append(result)
                else:
                    new_failed.append(fac_id)
                    
            except Exception as e:
                tqdm.write(f"  Facility {fac_id} failed: {e}")
                new_failed.append(fac_id)
                
                # Rate limit backoff
                if 'quota' in str(e).lower() or 'rate' in str(e).lower():
                    time.sleep(60)
            
            # Checkpoint every 10 facilities
            if (i + 1) % 10 == 0:
                pd.DataFrame(year_results).to_parquet(cache_path, index=False)
                pd.DataFrame({fac_id_col: new_failed}).to_parquet(failed_path, index=False)
                tqdm.write(f"  [checkpoint] {len(year_results)} results saved")
        
        # Final save for year
        if year_results:
            year_df = pd.DataFrame(year_results)
            year_df.to_parquet(cache_path, index=False)
            all_results.append(year_df)
        
        pd.DataFrame({fac_id_col: new_failed}).to_parquet(failed_path, index=False)
        print(f"[{year}] Complete: {len(year_results)} valid, {len(new_failed)} failed")
    
    if not all_results:
        return pd.DataFrame()
    
    combined = pd.concat(all_results, ignore_index=True)
    
    # =========================================================================
    # SIGNIFICANCE FLAGS (Beirle Sect. 3.11)
    # We do NOT hard-filter; instead we add boolean flags for downstream analysis
    # to use in sensitivity analyses. This follows the principle that we skip
    # Beirle's automatic point-source identification (we have known ETS/LCP
    # locations) but apply their quantification and significance logic.
    # =========================================================================
    
    # Detection limit flags (Beirle Sect. 3.11.1)
    # Conservative (0.11 kg/s) is appropriate for most European locations.
    # Permissive (0.03 kg/s) only valid for high-albedo desert conditions.
    combined['above_dl_0_11'] = combined['beirle_nox_kg_s'] >= DETECTION_LIMIT_0_11
    combined['above_dl_0_03'] = combined['beirle_nox_kg_s'] >= DETECTION_LIMIT_0_03
    
    # Statistical integration error flag (Beirle requires < 30%)
    combined['rel_err_stat_lt_0_3'] = combined['rel_err_stat'] < MAX_REL_ERR_STAT
    
    n_total = len(combined)
    n_above_0_11 = combined['above_dl_0_11'].sum()
    n_above_0_03 = combined['above_dl_0_03'].sum()
    n_stat_ok = combined['rel_err_stat_lt_0_3'].sum()
    print(f"Significance flags summary ({n_total} total):")
    print(f"  - above_dl_0_11: {n_above_0_11} ({100*n_above_0_11/n_total:.1f}%)")
    print(f"  - above_dl_0_03: {n_above_0_03} ({100*n_above_0_03/n_total:.1f}%)")
    print(f"  - rel_err_stat < 30%: {n_stat_ok} ({100*n_stat_ok/n_total:.1f}%)")
    
    return combined # type: ignore


def compute_interference_flags(
    beirle_panel: DataFrame,
    facilities_static: DataFrame,
    fac_id_col: str = 'idx',
    radius_km: float = INTERFERENCE_RADIUS_KM
) -> DataFrame:
    """
    Add interference flag based on proximity to other ETS/LCP facilities.
    
    For each facility, checks if there is another facility within `radius_km`.
    If so, the satellite outcome may reflect cluster-level rather than 
    single-facility emissions.
    
    Uses WGS84 distance approximation (same as used in 500m spatial clustering).
    
    Args:
        beirle_panel: Beirle panel DataFrame with idx column
        facilities_static: Static facility DataFrame with lat, lon, fac_id_col
        fac_id_col: Name of facility ID column
        radius_km: Interference radius in km (default: 20 km per Beirle)
        
    Returns:
        beirle_panel with 'interfered_20km' boolean column added
    """
    from scipy.spatial import cKDTree  # type: ignore
    
    # Get coordinates for all facilities (not just those in panel)
    # because a facility outside the panel can still interfere
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
    
    # Build list of interfered facility IDs
    fac_ids = fac_coords[fac_id_col].values  # type: ignore
    interfered_ids_set: set = set()
    for i, j in pairs:
        interfered_ids_set.add(fac_ids[i])
        interfered_ids_set.add(fac_ids[j])
    interfered_ids = list(interfered_ids_set)
    
    # Add flag to Beirle panel
    beirle_panel = beirle_panel.copy()
    beirle_panel['interfered_20km'] = beirle_panel[fac_id_col].isin(interfered_ids)
    
    n_total = len(beirle_panel[fac_id_col].unique())
    n_interfered = len(beirle_panel[beirle_panel['interfered_20km']][fac_id_col].unique()) # type: ignore
    print(f"Interference flag ({radius_km} km): {n_interfered}/{n_total} facilities "
          f"({100*n_interfered/n_total:.1f}%) have another ETS facility within radius")
    
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
    month: int = 6,
) -> None:
    """
    Plot mean advection map for a facility (for diagnostic visualization).
    
    This requires running the heavy GEE computation and is intended for
    debugging/exploration, not batch processing.
    """
    import matplotlib.pyplot as plt
    
    _ensure_ee_initialized()
    
    # Get data for the month
    start = ee.Date.fromYMD(year, month, 1) # type: ignore
    end = start.advance(1, 'month')
    
    no2_ic = (
        ee.ImageCollection(TROPOMI_NO2_COLLECTION) # type: ignore
        .filterDate(start, end)
        .select(TROPOMI_NO2_BAND)
    )
    
    # Mean NO₂ for the month
    no2_mean = no2_ic.mean()
    
    # Get wind
    daily_wind = _get_era5_daily_wind(year, month)
    wind_mean = daily_wind.mean()
    
    # Sample mean wind at location
    pt = ee.Geometry.Point([lon, lat]) # type: ignore
    wind_sample = wind_mean.reduceRegion(
        reducer=ee.Reducer.mean(), # type: ignore
        geometry=pt,
        scale=ERA5_SCALE_M
    ).getInfo()
    
    wind_u = wind_sample.get(ERA5_U_BAND, 0) # type: ignore
    wind_v = wind_sample.get(ERA5_V_BAND, 0) # type: ignore
    wind_speed = np.sqrt(wind_u**2 + wind_v**2)
    
    # Compute gradient
    gradient = no2_mean.gradient()
    dx = gradient.select('x').divide(TROPOMI_SCALE_M)
    dy = gradient.select('y').divide(TROPOMI_SCALE_M)
    
    # Advection
    advection = dx.multiply(wind_u).add(dy.multiply(wind_v))
    
    # Sample to array
    disc = _build_integration_disc(lon, lat, INTEGRATION_RADIUS_M * 1.5)
    
    print(f"Mean wind: {wind_speed:.1f} m/s ({wind_u:.1f}, {wind_v:.1f})")
    print("Fetching advection map (this may take a minute)...")
    
    # This is a simplified visualization - full implementation would
    # require proper array export which is slow in GEE
    
    import geemap  # type: ignore
    
    m = geemap.Map(center=[lat, lon], zoom=10)
    
    vis_params = {
        'min': -1e-9,
        'max': 1e-9,
        'palette': ['blue', 'white', 'red']
    }
    
    m.addLayer(advection.clip(disc), vis_params, 'Advection')
    m.addLayer(pt.buffer(500), {'color': 'black'}, 'Facility')
    
    return m
