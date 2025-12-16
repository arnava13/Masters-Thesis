"""
embedding_reduction.py
======================
Dimensionality reduction for AlphaEarth satellite embeddings.

Provides two strategies:
1. Standard PCA (unsupervised) - maximizes variance in embedding space
2. Facility-level PLS (supervised) - projects onto directions predictive of mean NOx

The facility-level PLS approach trains on cross-sectional facility means rather than
panel observations. This prevents outcome snooping / overfitting bias that would arise
from training on the same panel used for causal inference. See Chernozhukov et al. (2018)
for regularization bias, and Cinelli et al. (2022) on bad controls.

Key insight: By training PLS on facility-level means (one row per facility), the learned
projection is time-invariant and orthogonal to within-facility treatment variation.
This makes the reduced embeddings equivalent to pre-treatment covariates.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Union, overload
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Default column names
_DEFAULT_FAC_ID_COL = "idx"
_DEFAULT_YEAR_COL = "year"
_DEFAULT_EMB_PREFIX = "emb_"


# =============================================================================
# PCA Reduction (Unsupervised)
# =============================================================================

def reduce_embeddings_pca(
    df: pd.DataFrame,
    n_components: int = 10,
    emb_prefix: str = _DEFAULT_EMB_PREFIX,
    id_col: str = _DEFAULT_FAC_ID_COL,
    output_prefix: str = "pca_emb_",
    return_model: bool = False
) -> pd.DataFrame | Tuple[pd.DataFrame, PCA, StandardScaler]:
    """
    Apply standard PCA to embedding columns.
    
    Unsupervised dimensionality reduction that maximizes variance in embedding
    space. Safe for causal inference as it does not use outcome information.
    
    Parameters
    ----------
    df : DataFrame
        Panel data with embedding columns (emb_00, emb_01, ..., emb_63)
    n_components : int
        Number of PCA components to retain (default: 10)
    emb_prefix : str
        Prefix identifying embedding columns
    id_col : str
        Facility ID column
    output_prefix : str
        Prefix for output columns (default: 'pca_emb_')
    return_model : bool
        If True, also return fitted PCA and scaler objects
    
    Returns
    -------
    DataFrame with original columns plus PCA-reduced embedding columns.
    If return_model=True, also returns (pca, scaler) tuple.
    """
    df = df.copy()
    
    # Identify embedding columns
    emb_cols = [c for c in df.columns if c.startswith(emb_prefix)]
    if not emb_cols:
        raise ValueError(f"No columns found with prefix '{emb_prefix}'")
    
    n_orig = len(emb_cols)
    if n_components > n_orig:
        raise ValueError(f"n_components ({n_components}) > n_embeddings ({n_orig})")
    
    # Get embedding matrix (drop rows with missing embeddings)
    X = df[emb_cols].values
    valid_mask = ~np.isnan(X).any(axis=1)
    X_valid = X[valid_mask]
    
    if len(X_valid) == 0:
        raise ValueError("No valid embedding observations after dropping NaN")
    
    # Standardize before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)
    
    # Create output columns
    pca_col_names = [f"{output_prefix}{i:02d}" for i in range(n_components)]
    
    # Initialize with NaN
    for col in pca_col_names:
        df[col] = np.nan
    
    # Fill in transformed values
    df.loc[valid_mask, pca_col_names] = X_reduced
    
    # Report
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"PCA reduction: {n_orig} → {n_components} dimensions")
    print(f"  Variance explained: {var_explained:.1%}")
    print(f"  Valid observations: {valid_mask.sum()} / {len(df)}")
    
    if return_model:
        return df, pca, scaler
    return df


# =============================================================================
# Facility-Level PLS Reduction (Supervised, Causally Safe)
# =============================================================================

def reduce_embeddings_pls(
    df: pd.DataFrame,
    target_col: str,
    n_components: int = 5,
    emb_prefix: str = _DEFAULT_EMB_PREFIX,
    id_col: str = _DEFAULT_FAC_ID_COL,
    output_prefix: str = "pls_emb_",
    return_model: bool = False
) -> pd.DataFrame | Tuple[pd.DataFrame, PLSRegression, StandardScaler]:
    """
    Apply facility-level PLS to project embeddings onto outcome-relevant directions.
    
    PLS is trained on FACILITY-LEVEL MEANS (one row per facility), not
    panel observations. This prevents outcome snooping:
    - Panel PLS would learn from yearly shocks → biased treatment effects
    - Facility-level PLS learns only static geographic context → safe
    
    The resulting components are time-invariant within each facility, making them
    equivalent to pre-treatment covariates for causal identification.
    
    References
    ----------
    - Chernozhukov et al. (2018): Double/debiased ML; regularization bias
    - Cinelli et al. (2022): Bad controls in causal inference
    
    Parameters
    ----------
    df : DataFrame
        Panel data with embedding and target columns
    target_col : str
        Outcome column for supervised projection (e.g., 'beirle_nox_kg_s').
        PLS will be trained on facility-level MEAN of this variable.
    n_components : int
        Number of PLS components (default: 5)
    emb_prefix : str
        Prefix identifying embedding columns
    id_col : str
        Facility ID column
    output_prefix : str
        Prefix for output columns (default: 'pls_emb_')
    return_model : bool
        If True, also return fitted PLS and scaler objects
    
    Returns
    -------
    DataFrame with original columns plus PLS-reduced embedding columns.
    If return_model=True, also returns (pls, scaler) tuple.
    """
    df = df.copy()
    
    # Identify embedding columns
    emb_cols = [c for c in df.columns if c.startswith(emb_prefix)]
    if not emb_cols:
        raise ValueError(f"No columns found with prefix '{emb_prefix}'")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    n_orig = len(emb_cols)
    if n_components > n_orig:
        raise ValueError(f"n_components ({n_components}) > n_embeddings ({n_orig})")
    
    # =========================================================================
    # Step 1: Collapse to facility level (CRITICAL for causal validity)
    # =========================================================================
    # Average embeddings across years (should be constant, but average handles edge cases)
    fac_emb = df.groupby(id_col)[emb_cols].mean()
    
    # Average target across years (this is what PLS predicts)
    fac_target = df.groupby(id_col)[target_col].mean()
    
    # Align and drop missing
    valid_facs = fac_emb.index.intersection(fac_target.dropna().index)
    fac_emb_valid = fac_emb.loc[valid_facs]
    fac_target_valid = fac_target.loc[valid_facs]
    
    # Drop facilities with missing embeddings
    emb_valid_mask = ~fac_emb_valid.isna().any(axis=1)
    fac_emb_valid = fac_emb_valid[emb_valid_mask]
    fac_target_valid = fac_target_valid[emb_valid_mask]
    
    n_fac = len(fac_emb_valid)
    if n_fac < n_components:
        raise ValueError(f"Too few valid facilities ({n_fac}) for {n_components} PLS components")
    
    print(f"PLS training: {n_fac} facilities with valid embeddings + target")
    
    # =========================================================================
    # Step 2: Fit PLS on facility-level data
    # =========================================================================
    X_fac = fac_emb_valid.values
    y_fac = fac_target_valid.values.reshape(-1, 1)
    
    # Standardize
    scaler = StandardScaler()
    X_fac_scaled = scaler.fit_transform(X_fac)
    
    # Fit PLS
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_fac_scaled, y_fac)
    
    # =========================================================================
    # Step 3: Transform PANEL embeddings using facility-level projection
    # =========================================================================
    # Get panel-level embeddings
    X_panel = df[emb_cols].values
    panel_valid_mask = ~np.isnan(X_panel).any(axis=1)
    X_panel_valid = X_panel[panel_valid_mask]
    
    # Apply same scaler and PLS transform
    X_panel_scaled = scaler.transform(X_panel_valid)
    X_panel_reduced = pls.transform(X_panel_scaled)
    
    # Create output columns
    pls_col_names = [f"{output_prefix}{i:02d}" for i in range(n_components)]
    
    # Initialize with NaN
    for col in pls_col_names:
        df[col] = np.nan
    
    # Fill in transformed values
    df.loc[panel_valid_mask, pls_col_names] = X_panel_reduced
    
    # Report
    # PLS doesn't have explained_variance_ratio_ like PCA, use R² on training
    y_pred = pls.predict(X_fac_scaled)
    ss_res = np.sum((y_fac - y_pred) ** 2)
    ss_tot = np.sum((y_fac - y_fac.mean()) ** 2)
    r2_train = 1 - ss_res / ss_tot
    
    print(f"PLS reduction: {n_orig} → {n_components} dimensions")
    print(f"  Training R² (facility-level mean {target_col}): {r2_train:.3f}")
    print(f"  Facilities used for training: {n_fac}")
    print(f"  Panel observations with valid embeddings: {panel_valid_mask.sum()} / {len(df)}")
    
    if return_model:
        return df, pls, scaler
    return df


# =============================================================================
# Unified Interface
# =============================================================================

def reduce_embeddings(
    df: pd.DataFrame,
    method: str = "pca",
    n_components: int = 10,
    target_col: Optional[str] = None,
    emb_prefix: str = _DEFAULT_EMB_PREFIX,
    id_col: str = _DEFAULT_FAC_ID_COL,
    output_prefix: Optional[str] = None
) -> pd.DataFrame:  # type: ignore[return]
    """
    Unified interface for embedding dimensionality reduction.
    
    Parameters
    ----------
    df : DataFrame
        Panel data with embedding columns
    method : str
        'pca' for unsupervised PCA, 'pls' for facility-level supervised PLS
    n_components : int
        Number of components to retain
    target_col : str, optional
        Required for PLS; outcome column for supervised projection
    emb_prefix : str
        Prefix identifying embedding columns
    id_col : str
        Facility ID column
    output_prefix : str, optional
        Prefix for output columns (defaults to method-based prefix)
    
    Returns
    -------
    DataFrame with reduced embedding columns added
    """
    if output_prefix is None:
        output_prefix = f"{method}_emb_"
    
    if method == "pca":
        return reduce_embeddings_pca(
            df, n_components=n_components, emb_prefix=emb_prefix,
            id_col=id_col, output_prefix=output_prefix
        ) # type: ignore
    elif method == "pls":
        if target_col is None:
            raise ValueError("target_col required for PLS reduction")
        return reduce_embeddings_pls(
            df, target_col=target_col, n_components=n_components,
            emb_prefix=emb_prefix, id_col=id_col, output_prefix=output_prefix
        ) # type: ignore
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca' or 'pls'.")


def get_reduced_embedding_cols(
    df: pd.DataFrame,
    method: str = "pca",
    prefix: Optional[str] = None
) -> List[str]:
    """
    Get list of reduced embedding column names.
    
    Parameters
    ----------
    df : DataFrame
        Panel with reduced embedding columns
    method : str
        'pca' or 'pls'
    prefix : str, optional
        Custom prefix (defaults to method-based)
    
    Returns
    -------
    List of column names
    """
    if prefix is None:
        prefix = f"{method}_emb_"
    return [c for c in df.columns if c.startswith(prefix)]


def compare_reduction_methods(
    df: pd.DataFrame,
    target_col: str,
    n_components_list: List[int] = [5, 10, 15],
    emb_prefix: str = _DEFAULT_EMB_PREFIX,
    id_col: str = _DEFAULT_FAC_ID_COL
) -> pd.DataFrame:
    """
    Compare PCA and PLS reduction across different component counts.
    
    Returns summary table with variance explained (PCA) and R² (PLS).
    """
    results = []
    
    for n_comp in n_components_list:
        # PCA
        try:
            _, pca, _ = reduce_embeddings_pca(
                df, n_components=n_comp, emb_prefix=emb_prefix,
                id_col=id_col, return_model=True
            )
            var_exp = pca.explained_variance_ratio_.sum()
            results.append({
                "method": "PCA",
                "n_components": n_comp,
                "metric": "variance_explained",
                "value": var_exp
            })
        except Exception as e:
            print(f"PCA {n_comp} failed: {e}")
        
        # PLS
        try:
            _, pls, scaler = reduce_embeddings_pls(
                df, target_col=target_col, n_components=n_comp,
                emb_prefix=emb_prefix, id_col=id_col, return_model=True
            )
            # Compute R² on facility means
            emb_cols = [c for c in df.columns if c.startswith(emb_prefix)]
            fac_emb = df.groupby(id_col)[emb_cols].mean()
            fac_target = df.groupby(id_col)[target_col].mean()
            valid = fac_emb.index.intersection(fac_target.dropna().index)
            fac_emb = fac_emb.loc[valid].dropna()
            fac_target = fac_target.loc[fac_emb.index]
            
            X = scaler.transform(fac_emb.values)
            y = fac_target.values.reshape(-1, 1)
            y_pred = pls.predict(X)
            r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
            
            results.append({
                "method": "PLS",
                "n_components": n_comp,
                "metric": "r2_facility_mean",
                "value": r2
            })
        except Exception as e:
            print(f"PLS {n_comp} failed: {e}")
    
    return pd.DataFrame(results)


# =============================================================================
# Visualization & Interpretation
# =============================================================================

def plot_component_map(
    df: pd.DataFrame,
    component_idx: int = 0,
    method: str = "pca",
    prefix: Optional[str] = None,
    lat_col: str = "lat",
    lon_col: str = "lon",
    id_col: str = _DEFAULT_FAC_ID_COL,
    figsize: Tuple[int, int] = (14, 10),
    cmap: str = "RdBu_r"
) -> plt.Figure: # type: ignore
    """
    Map facilities colored by a PCA/PLS component value.
    
    Helps interpret what geographic patterns each component captures
    (e.g., North/South gradient, coastal vs inland, urban clusters).
    
    Parameters
    ----------
    df : DataFrame
        Panel with lat, lon, and reduced embedding columns
    component_idx : int
        Which component to visualize (0 = first component)
    method : str
        'pca' or 'pls'
    prefix : str, optional
        Custom column prefix
    lat_col, lon_col : str
        Coordinate columns
    id_col : str
        Facility ID column
    figsize : tuple
        Figure size
    cmap : str
        Colormap name
    
    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt
    
    if prefix is None:
        prefix = f"{method}_emb_"
    
    comp_col = f"{prefix}{component_idx:02d}"
    if comp_col not in df.columns:
        raise ValueError(f"Component column {comp_col} not found")
    
    # Collapse to facility level
    fac_df = df.groupby(id_col).agg({
        lat_col: "first",
        lon_col: "first", 
        comp_col: "mean"
    }).dropna()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    scatter = ax.scatter(
        fac_df[lon_col], fac_df[lat_col],
        c=fac_df[comp_col], cmap=cmap,
        s=50, alpha=0.7, edgecolors="k", linewidths=0.3
    )
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
    cbar.set_label(f"{method.upper()} Component {component_idx + 1}")
    
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Facility Locations by {method.upper()} Component {component_idx + 1}")
    
    plt.tight_layout()
    return fig


def plot_component_correlations(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_components: int = 10,
    method: str = "pca",
    prefix: Optional[str] = None,
    id_col: str = _DEFAULT_FAC_ID_COL,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure: # type: ignore
    """
    Correlate PCA/PLS components with interpretable facility features.
    
    Creates a heatmap showing correlations between embedding components
    and features like urban/rural, electricity sector, country dummies.
    
    Parameters
    ----------
    df : DataFrame
        Panel with reduced embedding columns and feature columns
    feature_cols : list
        Columns to correlate (e.g., ['in_urban_area', 'is_electricity'])
    n_components : int
        Number of components to include
    method : str
        'pca' or 'pls'
    prefix : str, optional
        Custom column prefix
    id_col : str
        Facility ID column
    figsize : tuple
        Figure size
    
    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if prefix is None:
        prefix = f"{method}_emb_"
    
    comp_cols = [f"{prefix}{i:02d}" for i in range(n_components)]
    comp_cols = [c for c in comp_cols if c in df.columns]
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    if not comp_cols or not feature_cols:
        raise ValueError("No valid component or feature columns found")
    
    # Collapse to facility level
    fac_df = df.groupby(id_col)[comp_cols + feature_cols].mean()
    
    # Compute correlations
    corr_matrix = []
    for feat in feature_cols:
        row = []
        for comp in comp_cols:
            valid = fac_df[[feat, comp]].dropna() # type: ignore
            if len(valid) > 2:
                corr = valid[feat].corr(valid[comp]) # type: ignore
            else:
                corr = np.nan
            row.append(corr)
        corr_matrix.append(row)
    
    corr_df = pd.DataFrame(
        corr_matrix, 
        index=feature_cols, # type: ignore
        columns=[f"PC{i+1}" for i in range(len(comp_cols))] # type: ignore
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title(f"{method.upper()} Components vs Interpretable Features")
    ax.set_xlabel("Component")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    return fig


def plot_component_grid(
    df: pd.DataFrame,
    n_components: int = 4,
    method: str = "pca",
    prefix: Optional[str] = None,
    lat_col: str = "lat",
    lon_col: str = "lon",
    id_col: str = _DEFAULT_FAC_ID_COL,
    figsize: Tuple[int, int] = (16, 12),
    cmap: str = "RdBu_r",
    europe_only: bool = True,
    add_basemap: bool = True
) -> plt.Figure: # type: ignore
    """
    Create a grid of maps showing multiple PCA/PLS components.
    
    Useful for comparing spatial patterns across components.
    
    Parameters
    ----------
    df : DataFrame
        Panel with lat, lon, and reduced embedding columns
    n_components : int
        Number of components to show
    method : str
        'pca' or 'pls'
    prefix : str, optional
        Custom column prefix
    lat_col, lon_col : str
        Coordinate columns
    id_col : str
        Facility ID column
    figsize : tuple
        Figure size
    cmap : str
        Colormap name
    europe_only : bool
        If True, zoom to Europe (lat > 30, lon > -30)
    add_basemap : bool
        If True, attempt to add country borders
    
    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt
    
    if prefix is None:
        prefix = f"{method}_emb_"
    
    # Collapse to facility level
    comp_cols = [f"{prefix}{i:02d}" for i in range(n_components)]
    comp_cols = [c for c in comp_cols if c in df.columns]
    n_show = len(comp_cols)
    
    if n_show == 0:
        raise ValueError(f"No component columns found with prefix {prefix}")
    
    fac_df = df.groupby(id_col).agg({
        lat_col: "first",
        lon_col: "first",
        **{c: "mean" for c in comp_cols}
    }).dropna()
    
    # Filter to Europe if requested
    if europe_only:
        fac_df = fac_df[(fac_df[lat_col] > 30) & (fac_df[lon_col] > -30)]
    
    ncols = 2
    nrows = (n_show + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_show > 1 else [axes]
    
    # Try to load Europe boundaries for basemap
    europe_borders = None
    if add_basemap:
        try:
            import geopandas as gpd
            # Use naturalearth_lowres from geopandas-datasets or download directly
            try:
                from geodatasets import get_path # type: ignore
                world = gpd.read_file(get_path('naturalearth.land'))
            except ImportError:
                # Fallback to direct URL
                url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
                world = gpd.read_file(url)
            europe_borders = world[world.geometry.centroid.y > 30]  # type: ignore
        except Exception:
            pass  # Silently fail if geopandas not available
    
    for i, comp_col in enumerate(comp_cols):
        ax = axes[i]
        
        # Add basemap if available
        if europe_borders is not None:
            europe_borders.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5)
        
        scatter = ax.scatter(
            fac_df[lon_col], fac_df[lat_col],
            c=fac_df[comp_col], cmap=cmap,
            s=40, alpha=0.8, edgecolors="k", linewidths=0.3,
            zorder=5  # Ensure points are on top
        )
        plt.colorbar(scatter, ax=ax, shrink=0.7)
        ax.set_title(f"{method.upper()} Component {i+1}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        # Set Europe bounds
        if europe_only:
            ax.set_xlim(-12, 35)
            ax.set_ylim(34, 62)
    
    # Hide unused axes
    for j in range(n_show, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(f"Geographic Distribution of {method.upper()} Components", fontsize=14)
    plt.tight_layout()
    return fig
