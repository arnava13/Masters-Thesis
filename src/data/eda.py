"""
EDA (Exploratory Data Analysis) functions for facilities and ETS data.

Contains:
- fac_eda: EDA for facilities static and yearly panels
- ets_eda: EDA for ETS stringency/allocation data
- beirle_eda: EDA for satellite NOx emission proxy
"""

import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from typing import Optional


def fac_eda(
    static_panel: DataFrame,
    yearly_panel: DataFrame,
    fac_id_col: str = 'idx',
    n_samples: int = 6,
    nuts_geojson_path: str = "",
    pypsa_clusters_path: str = "",
    pypsa_resolution: str = "128",
) -> None:
    """
    EDA for facilities (static) and yearly panel (time-varying capacity + fuel).
    
    Args:
        static_panel: facilities_static DataFrame
        yearly_panel: facilities_yearly DataFrame
        fac_id_col: Name of facility ID column (default: 'idx')
        n_samples: Number of sample facilities to plot (default: 6)
        nuts_geojson_path: Path to NUTS2 GeoJSON for regional maps
        pypsa_clusters_path: Path to PyPSA cluster shapefile
        pypsa_resolution: PyPSA cluster resolution (default: '128')
    """
    d = static_panel.copy()

    if not yearly_panel.empty:
        
        # Fuel share statistics from yearly panel
        share_cols = [c for c in yearly_panel.columns if c.startswith('share_')]
        if share_cols:
            # Fuel mix change over time (aggregate)
            print("\nFuel Mix Evolution Over Time:")
            yearly_means = yearly_panel.groupby('year')[share_cols].mean()
            yearly_means.columns = [c.replace('share_', '').replace('_', ' ').title() for c in yearly_means.columns]  
            
            fig, ax = plt.subplots(figsize=(10, 5))
            yearly_means.plot(kind='bar', stacked=True, ax=ax, width=0.8)
            ax.set_xlabel('Year')
            ax.set_ylabel('Average Share')
            ax.set_title('Average Fuel Mix by Year')
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
        
        # ------------------------------------------------------------------
        # Sample facility time series: Capacity
        # ------------------------------------------------------------------
        print("\n" + "="*60)
        print("Sample Facility Time Series")
        print("="*60)
        
        all_facs = yearly_panel[fac_id_col].unique()
        n_plot = min(n_samples, len(all_facs))
        sample_facs = np.random.choice(all_facs, size=n_plot, replace=False) # type: ignore
        
        # Capacity time series
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        axes = axes.flatten()
        for i, fac_idx in enumerate(sample_facs):
            ax = axes[i]
            fac_data = yearly_panel[yearly_panel[fac_id_col] == fac_idx].sort_values('year') # type: ignore
           
            ax.plot(fac_data['year'], fac_data['capacity_mw'], 'o-', color='steelblue', linewidth=2, markersize=6)
            ax.fill_between(fac_data['year'], 0, fac_data['capacity_mw'], alpha=0.2, color='steelblue')
            ax.set_title(f"{fac_idx}", fontsize=9)
            ax.set_xlabel('Year')
            ax.set_ylabel('Capacity (MW)')
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # type: ignore
            ax.grid(alpha=0.3)
        
        plt.suptitle('Sample Facilities: Capacity Over Time', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Fuel mix time series (stacked area)
        if share_cols:
            fig, axes = plt.subplots(2, 3, figsize=(14, 7))
            axes = axes.flatten()
            
            # Clean column names for legend
            share_labels = [c.replace('share_', '').replace('_', ' ').title() for c in share_cols]
            colors = plt.cm.Set2(np.linspace(0, 1, len(share_cols))) # type: ignore
            
            for i, fac_idx in enumerate(sample_facs):
                ax = axes[i]
                fac_data = yearly_panel[yearly_panel[fac_id_col] == fac_idx].sort_values('year') # type: ignore
                
                # Stacked area chart (only pass labels for first subplot)
                stack_kwargs = {"colors": colors, "alpha": 0.8}
                if i == 0:
                    stack_kwargs["labels"] = share_labels
                ax.stackplot(fac_data['year'], 
                            [fac_data[c].values for c in share_cols],
                            **stack_kwargs)
                ax.set_title(f"{fac_idx}", fontsize=9)
                ax.set_xlabel('Year')
                ax.set_ylabel('Fuel Share')
                ax.set_ylim(0, 1)
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # type: ignore
                ax.grid(alpha=0.3)
            
            # Single legend for all subplots
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), 
                      ncol=len(share_cols), fontsize=9)
            
            plt.suptitle('Sample Facilities: Fuel Mix Over Time', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.12)
            plt.show()
    
    # Country distribution
    country_counts = d.groupby('country_code').size().reset_index(name='n')  # type: ignore
    country_counts['share'] = country_counts['n'] / country_counts['n'].sum()
    from IPython.display import display
    display(country_counts.sort_values('n', ascending=False).head(12))
    
    # ------------------------------------------------------------------
    # Urbanization Analysis
    # ------------------------------------------------------------------
    if 'urbanization_degree' in d.columns or 'in_urban_area' in d.columns:
        print("\n" + "="*60)
        print("Urbanization Analysis")
        print("="*60)
        
        if 'urbanization_degree' in d.columns:
            # SMOD class distribution
            smod_labels = {
                10: "Water", 11: "Very low density rural", 12: "Low density rural",
                13: "Rural cluster", 21: "Suburban", 22: "Semi-dense urban",
                23: "Dense urban", 30: "Urban centre"
            }
            smod_counts = d['urbanization_degree'].value_counts().sort_index()
            smod_df = pd.DataFrame({
                'SMOD Code': smod_counts.index,
                'Label': [smod_labels.get(int(x), f"Unknown ({x})") for x in smod_counts.index],
                'Count': smod_counts.values,
                'Share': (smod_counts.values / smod_counts.sum() * 100).round(1)
            })
            print("\nUrbanization Degree (GHSL SMOD) Distribution:")
            display(smod_df)
            
            # Bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(smod_counts)))  # type: ignore
            bars = ax.bar(range(len(smod_counts)), smod_counts.values, color=colors, edgecolor='black') # type: ignore
            ax.set_xticks(range(len(smod_counts)))
            ax.set_xticklabels([f"{int(x)}\n{smod_labels.get(int(x), '')[:12]}" for x in smod_counts.index], 
                             fontsize=8, rotation=0)
            ax.set_xlabel('SMOD Class')
            ax.set_ylabel('Number of Facilities')
            ax.set_title('Facility Distribution by Urbanization Degree')
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        if 'in_urban_area' in d.columns:
            n_urban = d['in_urban_area'].sum()
            n_rural = len(d) - n_urban
            print(f"\nUrban/Rural Split (SMOD ≥21 = Urban):")
            print(f"  Urban: {n_urban:,} ({100*n_urban/len(d):.1f}%)")
            print(f"  Rural: {n_rural:,} ({100*n_rural/len(d):.1f}%)")
    
    # ------------------------------------------------------------------
    # Spatial Clustering Maps (NUTS2 and PyPSA)
    # ------------------------------------------------------------------
    _plot_spatial_clustering_maps(
        d, fac_id_col, nuts_geojson_path, pypsa_clusters_path, pypsa_resolution
    )


def _plot_spatial_clustering_maps(
    static_panel: DataFrame,
    fac_id_col: str,
    nuts_geojson_path: str,
    pypsa_clusters_path: str,
    pypsa_resolution: str,
) -> None:
    """Plot spatial clustering maps for NUTS2 and PyPSA regions."""
    try:
        import geopandas as gpd  # type: ignore
    except ImportError:
        print("\nSkipping spatial maps (geopandas not installed)")
        return
    
    d = static_panel.copy()
    
    # ------------------------------------------------------------------
    # NUTS2 Regional Map (All Facilities)
    # ------------------------------------------------------------------
    if 'nuts2_region' in d.columns and nuts_geojson_path:
        print("\n" + "="*60)
        print("NUTS2 Regional Distribution (All Facilities)")
        print("="*60)
        
        try:
            nuts_gdf = gpd.read_file(nuts_geojson_path)
            
            # Count facilities per NUTS2 region
            nuts_counts = d['nuts2_region'].value_counts().reset_index()
            nuts_counts.columns = ['NUTS_ID', 'n_facilities']
            
            # Merge counts with geometry
            nuts_gdf = nuts_gdf.merge(nuts_counts, on='NUTS_ID', how='left')
            nuts_gdf['n_facilities'] = nuts_gdf['n_facilities'].fillna(0)
            
            # Filter to Europe extent
            nuts_gdf = nuts_gdf.cx[-25:45, 34:72]  # type: ignore
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Base map (all regions)
            nuts_gdf.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.3)
            
            # Choropleth for regions with facilities
            nuts_with_fac = nuts_gdf[nuts_gdf['n_facilities'] > 0]
            if len(nuts_with_fac) > 0:
                nuts_with_fac.plot(
                    column='n_facilities', ax=ax, legend=True,
                    cmap='YlOrRd', edgecolor='black', linewidth=0.5,
                    legend_kwds={'label': 'Number of Facilities', 'shrink': 0.6}
                )
            
            # Add facility points
            if 'lon' in d.columns and 'lat' in d.columns:
                ax.scatter(d['lon'], d['lat'], s=8, c='blue', alpha=0.5, 
                          edgecolor='none', label=f'Facilities (n={len(d)})')
            
            ax.set_xlim(-15, 35)
            ax.set_ylim(34, 72)
            ax.set_title(f'NUTS2 Regional Distribution\n{d["nuts2_region"].nunique()} regions, {len(d)} facilities')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.legend(loc='lower left')
            plt.tight_layout()
            plt.show()
            
            # Top regions table
            print("\nTop 10 NUTS2 Regions by Facility Count:")
            top_nuts = nuts_counts.nlargest(10, 'n_facilities')
            display(top_nuts)
            
        except Exception as e:
            print(f"Error plotting NUTS2 map: {e}")
    
    # ------------------------------------------------------------------
    # PyPSA Cluster Map (Electricity Facilities Only)
    # ------------------------------------------------------------------
    if 'pypsa_cluster' in d.columns and 'is_electricity' in d.columns:
        print("\n" + "="*60)
        print("PyPSA Cluster Distribution (Electricity Facilities Only)")
        print("="*60)
        
        # Filter to electricity facilities
        elec_facs = d[d['is_electricity'] == True].copy()
        n_elec = len(elec_facs)
        
        if n_elec == 0:
            print("No electricity facilities found")
            return
        
        print(f"Electricity facilities: {n_elec} / {len(d)} ({100*n_elec/len(d):.1f}%)")
        
        try:
            import os
            
            if not os.path.exists(pypsa_clusters_path):
                print(f"PyPSA clusters file not found: {pypsa_clusters_path}")
                return
            
            pypsa_gdf = gpd.read_file(pypsa_clusters_path)
            
            # Handle pypsa_cluster values: could be float indices (64.0) or string names ("DE0 0")
            cluster_values = elec_facs['pypsa_cluster'].dropna()  # type: ignore[union-attr]
            sample_val = cluster_values.iloc[0] if len(cluster_values) > 0 else None
            
            # Check if values are numeric (indices) or string names
            is_numeric_index = isinstance(sample_val, (int, float)) or (
                isinstance(sample_val, str) and sample_val.replace('.', '').replace('-', '').isdigit()
            )
            
            if is_numeric_index and 'name' in pypsa_gdf.columns:
                # Create index-to-name mapping from shapefile order
                pypsa_gdf = pypsa_gdf.reset_index(drop=True)
                idx_to_name = dict(enumerate(pypsa_gdf['name']))
                
                # Convert float indices to string names
                elec_facs = elec_facs.copy()
                elec_facs['pypsa_cluster_name'] = elec_facs['pypsa_cluster'].apply(  # type: ignore[union-attr]
                    lambda x: idx_to_name.get(int(x), None) if pd.notna(x) else None
                )
                cluster_col_to_use = 'pypsa_cluster_name'
                join_col = 'name'
                print(f"Converting numeric indices to names (e.g., {int(sample_val)} -> {idx_to_name.get(int(sample_val), 'N/A')})")  # type: ignore[arg-type]
            else:
                # Values are already string names
                cluster_col_to_use = 'pypsa_cluster'
                # Find the join column in the shapefile
                join_col = None
                for col in ['name', 'NAME', 'region', 'cluster', 'id']:
                    if col in pypsa_gdf.columns:
                        join_col = col
                        break
                if join_col is None:
                    print(f"Warning: Could not find cluster name column. Columns: {list(pypsa_gdf.columns)}")
                    join_col = pypsa_gdf.columns[0]
            
            # Count electricity facilities per PyPSA cluster
            pypsa_counts = elec_facs[cluster_col_to_use].value_counts().reset_index() # type: ignore
            pypsa_counts.columns = [join_col, 'n_facilities']
            
            # Merge counts with geometry
            pypsa_gdf = pypsa_gdf.merge(pypsa_counts, on=join_col, how='left')
            pypsa_gdf['n_facilities'] = pypsa_gdf['n_facilities'].fillna(0)
            
            n_matched = (pypsa_gdf['n_facilities'] > 0).sum()
            print(f"Matched {n_matched} clusters with facilities (join column: '{join_col}')")
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Base map (all clusters)
            pypsa_gdf.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.3)
            
            # Choropleth for clusters with electricity facilities
            pypsa_with_fac = pypsa_gdf[pypsa_gdf['n_facilities'] > 0]
            if len(pypsa_with_fac) > 0:
                pypsa_with_fac.plot(
                    column='n_facilities', ax=ax, legend=True,
                    cmap='YlGnBu', edgecolor='black', linewidth=0.5,
                    legend_kwds={'label': 'Number of Electricity Facilities', 'shrink': 0.6}
                )
            
            # Add electricity facility points
            if 'lon' in elec_facs.columns and 'lat' in elec_facs.columns:
                ax.scatter(elec_facs['lon'], elec_facs['lat'], s=12, c='red', alpha=0.6, 
                          edgecolor='black', linewidth=0.3, label=f'Electricity (n={n_elec})')
            
            ax.set_xlim(-15, 35)
            ax.set_ylim(34, 72)
            ax.set_title(f'PyPSA-Eur Cluster Distribution (Electricity Only)\n'
                        f'{elec_facs["pypsa_cluster"].nunique()} clusters, {n_elec} facilities') # type: ignore
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.legend(loc='lower left')
            plt.tight_layout()
            plt.show()
            
            # Top clusters table
            print("\nTop 10 PyPSA Clusters by Electricity Facility Count:")
            top_pypsa = pypsa_counts.nlargest(10, 'n_facilities')
            display(top_pypsa)
            
        except Exception as e:
            print(f"Error plotting PyPSA map: {e}")


def ets_eda(
    facilities_yearly: DataFrame,
    facilities_static: Optional[DataFrame] = None,
    fac_id_col: str = 'idx',
    n_samples: int = 6,
) -> None:
    """
    EDA for ETS stringency/allocation data.
    
    Args:
        facilities_yearly: facilities_yearly DataFrame with ETS columns
        facilities_static: Optional static DataFrame with is_electricity flag
        fac_id_col: Name of facility ID column (default: 'idx')
        n_samples: Number of sample facilities to plot (default: 6)
    """
    from IPython.display import display
    
    # ------------------------------------------------------------------
    # 1. Electricity Sector Analysis (if static panel provided)
    # ------------------------------------------------------------------
    if facilities_static is not None and 'is_electricity' in facilities_static.columns:
        print("\n" + "="*60)
        print("1. Electricity Sector Analysis")
        print("="*60)
        
        n_total = len(facilities_static)
        n_elec = facilities_static['is_electricity'].sum()
        n_other = n_total - n_elec
        
        print(f"\nFacility Classification (EU ETS Activity Codes):")
        print(f"  Electricity generators (codes 1, 20): {n_elec:,} ({100*n_elec/n_total:.1f}%)")
        print(f"  Other industrial:                     {n_other:,} ({100*n_other/n_total:.1f}%)")
        
        # Merge is_electricity into yearly for comparison
        yearly_with_elec = facilities_yearly.merge(
            facilities_static[[fac_id_col, 'is_electricity']], 
            on=fac_id_col, how='left'
        )
        
        if 'eu_verified_tco2' in yearly_with_elec.columns:
            # Emissions comparison
            elec_emissions = yearly_with_elec[yearly_with_elec['is_electricity'] == True]['eu_verified_tco2']
            other_emissions = yearly_with_elec[yearly_with_elec['is_electricity'] == False]['eu_verified_tco2']
            
            print(f"\nVerified Emissions (tCO₂/yr) by Sector:")
            print(f"  Electricity: mean={elec_emissions.mean()/1e6:.2f} MtCO₂, median={elec_emissions.median()/1e6:.2f} MtCO₂") # type: ignore
            print(f"  Other:       mean={other_emissions.mean()/1e6:.2f} MtCO₂, median={other_emissions.median()/1e6:.2f} MtCO₂") # type: ignore
            
            # Distribution comparison plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Log emissions histogram by sector
            ax = axes[0]
            elec_log = np.log10(elec_emissions.clip(lower=1)) # type: ignore
            other_log = np.log10(other_emissions.clip(lower=1)) # type: ignore
            ax.hist(elec_log.dropna(), bins=30, alpha=0.6, label=f'Electricity (n={len(elec_log)})', color='orange') # type: ignore
            ax.hist(other_log.dropna(), bins=30, alpha=0.6, label=f'Other (n={len(other_log)})', color='steelblue') # type: ignore
            ax.set_xlabel('log₁₀(Verified tCO₂)')
            ax.set_ylabel('Count')
            ax.set_title('Emissions Distribution by Sector')
            ax.legend()
            ax.grid(alpha=0.3)
            
            # Allocation ratio by sector
            ax = axes[1]
            if 'eu_alloc_ratio' in yearly_with_elec.columns:
                elec_ratio = yearly_with_elec[yearly_with_elec['is_electricity'] == True]['eu_alloc_ratio'].clip(0, 5)
                other_ratio = yearly_with_elec[yearly_with_elec['is_electricity'] == False]['eu_alloc_ratio'].clip(0, 5)
                ax.hist(elec_ratio.dropna(), bins=30, alpha=0.6, label='Electricity', color='orange') # type: ignore
                ax.hist(other_ratio.dropna(), bins=30, alpha=0.6, label='Other', color='steelblue') # type: ignore
                ax.axvline(1, color='black', linestyle='--', linewidth=1, label='Ratio = 1')
                ax.set_xlabel('Allocation Ratio')
                ax.set_ylabel('Count')
                ax.set_title('Allocation Ratio by Sector')
                ax.legend()
                ax.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Aggregate by year and sector
            print("\nAggregate Emissions by Year and Sector (MtCO₂):")
            yearly_sector = yearly_with_elec.groupby(['year', 'is_electricity'])['eu_verified_tco2'].sum() / 1e6 # type: ignore
            yearly_sector = yearly_sector.unstack().rename(columns={True: 'Electricity', False: 'Other'})
            display(yearly_sector.round(2))
    
    # ------------------------------------------------------------------
    # 2. Time series for sample plants
    # ------------------------------------------------------------------
    print("\n2. Time Series for Sample Plants (allocatedTotal):")
    all_plants = facilities_yearly[fac_id_col].unique()
    n_plot = min(n_samples, len(all_plants))
    sample_plants = np.random.choice(all_plants, size=n_plot, replace=False) # type: ignore

    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    for i, idx in enumerate(sample_plants):
        ax = axes[i // 3, i % 3]
        plant_data = facilities_yearly[facilities_yearly[fac_id_col] == idx].sort_values('year')  # type: ignore
        
        ax.plot(plant_data['year'], plant_data['eu_verified_tco2'] / 1e6, 'o-', label='Verified', markersize=4)
        ax.plot(plant_data['year'], plant_data['eu_alloc_total_tco2'] / 1e6, 's--', label='Alloc Total', markersize=4, alpha=0.7)
        ax.set_title(f"Facility {idx}", fontsize=9)
        ax.set_xlabel('Year')
        ax.set_ylabel('MtCO₂')
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # type: ignore
        if i == 0:
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.suptitle(f'Random Sample of {n_plot} Sites: Verified vs Total Allocated', fontsize=12)
    plt.tight_layout()
    plt.show()


def beirle_eda(
    beirle_panel: DataFrame,
    static_panel: DataFrame,
    facilities_yearly: Optional[DataFrame] = None,
    fac_id_col: str = "idx",
):
    """
    EDA for Beirle-style NOx emission estimates.
    
    Args:
        beirle_panel: Panel with NOx emissions and satellite data
        static_panel: Static facility attributes (country_code, etc.)
        facilities_yearly: Optional yearly panel with ETS data for correlation plot
        fac_id_col: Facility ID column name
    
    Key variables:
    - beirle_nox_kg_s: NOx emission rate (kg/s)
    - c_tau_mean: Mean lifetime correction factor
    - rel_err_total: Total relative uncertainty (as fraction)
    """
    if beirle_panel.empty:
        print("No Beirle panel data available")
        return
        
    df = beirle_panel.copy()
    
    # Merge country for grouping
    df = df.merge(static_panel[[fac_id_col, 'country_code']], on=fac_id_col, how='left')
    
    print("="*60)
    print("BEIRLE NOx EMISSION EDA")
    print("="*60)
    print(f"Facility-years: {len(df):,} | Facilities: {df[fac_id_col].nunique():,}")
    print(f"Years: {int(df['year'].min())}–{int(df['year'].max())}")
    
    # -------------------------------------------------------------------------
    # 1. Summary Statistics
    # -------------------------------------------------------------------------
    print("\n" + "-"*40)
    print("1. Summary Statistics")
    print("-"*40)
    
    emission_cols = ['beirle_nox_kg_s', 'c_tau_mean', 'rel_err_total', 'n_days_satellite']
    avail_cols = [c for c in emission_cols if c in df.columns]
    
    if avail_cols:
        stats = df[avail_cols].describe().T
        stats['non_null'] = df[avail_cols].notna().sum()
        display(stats.round(4))
    
    # -------------------------------------------------------------------------
    # 2. Distribution of Emissions
    # -------------------------------------------------------------------------
    print("\n" + "-"*40)
    print("2. Distribution of NOx Emissions")
    print("-"*40)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # NOx emission rate (kg/s)
    ax = axes[0, 0]
    if 'beirle_nox_kg_s' in df.columns:
        data = df['beirle_nox_kg_s'].dropna()
        ax.hist(data, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(0.01, color='red', linestyle='--', linewidth=2, label='DL = 0.01 kg/s')
        ax.axvline(data.mean(), color='green', linestyle='-', linewidth=2, 
                   label=f'Mean = {data.mean():.3f}')
        ax.set_xlabel('NOx Emission (kg/s)')
        ax.set_ylabel('Frequency')
        ax.set_title('NOx Emission Rate Distribution')
        ax.legend()
    
    # Lifetime correction factor (c_tau_mean)
    ax = axes[0, 1]
    if 'c_tau_mean' in df.columns:
        data = df['c_tau_mean'].dropna()
        ax.hist(data, bins=30, color='orange', edgecolor='black', alpha=0.7)
        ax.axvline(data.mean(), color='green', linestyle='-', linewidth=2,
                   label=f'Mean = {data.mean():.2f}')
        ax.set_xlabel('Mean Lifetime Correction Factor (c_τ)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of c_τ (expected 1.2-1.8)')
        ax.legend()
    
    # Relative uncertainty (pre-filtered to ≤50% in satellite_outcome.py)
    ax = axes[1, 0]
    if 'rel_err_total' in df.columns:
        data = df['rel_err_total'].dropna() * 100  # Convert to %
        ax.hist(data, bins=30, color='purple', edgecolor='black', alpha=0.7)
        ax.axvline(data.median(), color='green', linestyle='-', linewidth=2,
                   label=f'Median = {data.median():.1f}%')
        ax.set_xlabel('Total Relative Uncertainty (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Uncertainty Distribution (filtered ≤50%)')
        ax.legend()
    
    # Number of satellite observation days
    ax = axes[1, 1]
    if 'n_days_satellite' in df.columns:
        data = df['n_days_satellite'].dropna()
        ax.hist(data, bins=30, color='teal', edgecolor='black', alpha=0.7)
        ax.axvline(data.mean(), color='green', linestyle='-', linewidth=2,
                   label=f'Mean = {data.mean():.0f}')
        ax.set_xlabel('Number of Valid Satellite Days')
        ax.set_ylabel('Frequency')
        ax.set_title('Satellite Observation Coverage')
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # -------------------------------------------------------------------------
    # 3. Above Detection Limit
    # -------------------------------------------------------------------------
    print("\n" + "-"*40)
    print("3. Detection Limit Analysis")
    print("-"*40)
    
    if 'above_dl' in df.columns:
        above_dl = df['above_dl'].sum()
        pct_above = 100 * above_dl / len(df)
        print(f"Above DL (0.01 kg/s): {above_dl:,} ({pct_above:.1f}%)")
    else:
        print("No above_dl column found")
    
    # -------------------------------------------------------------------------
    # 4. By Country
    # -------------------------------------------------------------------------
    print("\n" + "-"*40)
    print("4. Emissions by Country")
    print("-"*40)

    country_means = df.groupby('country_code')['beirle_nox_kg_s'].mean().dropna()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    countries_sorted = country_means.sort_values(ascending=False) # type: ignore
    ax.bar(countries_sorted.index, countries_sorted.values, color='steelblue', edgecolor='black', alpha=0.7) # type: ignore
    ax.axhline(country_means.mean(), color='red', linestyle='--', linewidth=2, # type: ignore
            label=f'Overall Mean = {country_means.mean():.3f} kg/s')
    ax.set_xlabel('Country Code')
    ax.set_ylabel('Mean NOx Emissions (kg/s)')
    ax.set_title('Distribution of Mean NOx Emissions by Country')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    # -------------------------------------------------------------------------
    # 5. Reported NOx vs Satellite NOx Correlation (by Sample Split)
    # -------------------------------------------------------------------------
    print("\n" + "-"*40)
    print("5. Reported NOx vs Satellite NOx Correlation (by Sample Split)")
    print("-"*40)
    
    # Merge reported NOx and static attributes for splits
    if facilities_yearly is not None and 'reported_nox_tonnes' in facilities_yearly.columns:
        df_merged = df.merge(
            facilities_yearly[[fac_id_col, 'year', 'reported_nox_tonnes']],
            on=[fac_id_col, 'year'],
            how='left'
        )
    elif 'reported_nox_tonnes' in df.columns:
        df_merged = df
    else:
        print("Reported NOx data not available. Pass facilities_yearly with reported_nox_tonnes column.")
        return
    
    # Merge static attributes for urban/interference splits
    if static_panel is not None:
        static_cols = [fac_id_col]
        if 'in_urban_area' in static_panel.columns:
            static_cols.append('in_urban_area')
        if 'interfered_5km' in static_panel.columns:
            static_cols.append('interfered_5km')
        df_merged = df_merged.merge(static_panel[static_cols], on=fac_id_col, how='left')
    
    # Filter to above detection limit
    if 'above_dl' in df_merged.columns:
        df_merged = df_merged[df_merged['above_dl'] == True].copy()
    
    # Define sample splits: 1) All, 2) Non-urban, 3) Non-urban + not interfered
    sample_configs = [
        {'name': 'All Points', 'filter': lambda x: x},
        {'name': 'Non-Urban', 'filter': lambda x: x[x.get('in_urban_area', True) == False] if 'in_urban_area' in x.columns else x},
        {'name': 'Non-Urban + Not Interfered', 'filter': lambda x: x[(x.get('in_urban_area', True) == False) & (x.get('interfered_5km', True) == False)] if 'in_urban_area' in x.columns and 'interfered_5km' in x.columns else x},
    ]
    
    # Colors for the three samples
    sample_colors = ['#3498db', '#27ae60', '#e74c3c']  # blue, green, red
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (cfg, ax, color) in enumerate(zip(sample_configs, axes, sample_colors)):
        sample_name = cfg['name']
        try:
            sample_df = cfg['filter'](df_merged)
        except Exception:
            sample_df = df_merged
        
        plot_df = sample_df[['beirle_nox_kg_s', 'reported_nox_tonnes']].dropna() # type: ignore
        
        if len(plot_df) < 10:
            print(f"  {sample_name}: Insufficient data (n={len(plot_df)})")
            ax.set_title(f'{sample_name}\n(n={len(plot_df)}, insufficient)')
            ax.set_xlabel('Reported NOx (tonnes/yr)')
            ax.set_ylabel('Satellite NOx (kg/s)')
            continue
        
        # Compute correlations
        corr = plot_df['beirle_nox_kg_s'].corr(plot_df['reported_nox_tonnes'])  # type: ignore
        log_corr = np.log10(plot_df['beirle_nox_kg_s'].clip(lower=1e-6)).corr(  # type: ignore
            np.log10(plot_df['reported_nox_tonnes'].clip(lower=1))  # type: ignore
        )
        
        print(f"  {sample_name}: r={corr:.3f} (levels), r={log_corr:.3f} (log-log), n={len(plot_df):,}")
        
        # Scatterplot
        ax.scatter(plot_df['reported_nox_tonnes'], plot_df['beirle_nox_kg_s'],
                   alpha=0.6, s=25, color=color, edgecolor='none')
        ax.set_xlabel('Reported NOx (tonnes/yr)')
        ax.set_ylabel('Satellite NOx (kg/s)')
        ax.set_title(f'{sample_name}\nr={corr:.3f} (n={len(plot_df):,})')
        ax.grid(alpha=0.3)
    
    plt.suptitle('Reported vs Satellite NOx — Sample Splits (DL ≥ 0.01 kg/s)', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Log-log version
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (cfg, ax, color) in enumerate(zip(sample_configs, axes, sample_colors)):
        sample_name = cfg['name']
        try:
            sample_df = cfg['filter'](df_merged)
        except Exception:
            sample_df = df_merged
        
        plot_df = sample_df[['beirle_nox_kg_s', 'reported_nox_tonnes']].dropna() # type: ignore
        
        if len(plot_df) < 10:
            ax.set_title(f'{sample_name}\n(insufficient data)')
            ax.set_xlabel('log₁₀(Reported NOx)')
            ax.set_ylabel('log₁₀(Satellite NOx)')
            continue
        
        log_corr = np.log10(plot_df['beirle_nox_kg_s'].clip(lower=1e-6)).corr(  # type: ignore
            np.log10(plot_df['reported_nox_tonnes'].clip(lower=1))  # type: ignore
        )
        
        # Scatterplot (log-log)
        ax.scatter(np.log10(plot_df['reported_nox_tonnes'].clip(lower=1)),  # type: ignore
                   np.log10(plot_df['beirle_nox_kg_s'].clip(lower=1e-6)),  # type: ignore
                   alpha=0.6, s=25, color=color, edgecolor='none')
        ax.set_xlabel('log₁₀(Reported NOx)')
        ax.set_ylabel('log₁₀(Satellite NOx)')
        ax.set_title(f'{sample_name}\nr={log_corr:.3f} (n={len(plot_df):,})')
        ax.grid(alpha=0.3)
    
    plt.suptitle('Reported vs Satellite NOx — Log-Log Scale (DL ≥ 0.01 kg/s)', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()