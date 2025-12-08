"""
EDA (Exploratory Data Analysis) functions for facilities and ETS data.

Contains:
- fac_eda: EDA for facilities static and yearly panels
- ets_eda: EDA for ETS stringency/allocation data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def fac_eda(
    static_panel: pd.DataFrame,
    yearly_panel: pd.DataFrame,
    fac_id_col: str = 'idx',
    n_samples: int = 6,
) -> None:
    """
    EDA for facilities (static) and yearly panel (time-varying capacity + fuel).
    
    Args:
        static_panel: facilities_static DataFrame
        yearly_panel: facilities_yearly DataFrame
        fac_id_col: Name of facility ID column (default: 'idx')
        n_samples: Number of sample facilities to plot (default: 6)
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


def ets_eda(
    facilities_yearly: pd.DataFrame,
    fac_id_col: str = 'idx',
    n_samples: int = 6,
) -> None:
    """
    EDA for ETS stringency/allocation data.
    
    Args:
        facilities_yearly: facilities_yearly DataFrame with ETS columns
        fac_id_col: Name of facility ID column (default: 'idx')
        n_samples: Number of sample facilities to plot (default: 6)
    """
    from IPython.display import display
    
    # Basic stats
    print("\n1. Summary Statistics:")
    cols = ['eu_verified_tco2', 'eu_alloc_total_tco2', 'eu_shortfall_tco2', 'eu_alloc_ratio']
    display(facilities_yearly[cols].describe().round(2))

    # Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Verified emissions distribution (log scale)
    ax = axes[0, 0]
    facilities_yearly['eu_verified_tco2'].clip(lower=1).apply(np.log10).hist(bins=40, ax=ax, edgecolor='black', alpha=0.7)
    ax.set_xlabel('log₁₀(Verified tCO₂)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Verified Emissions')

    # Shortfall distribution (now using allocatedTotal)
    ax = axes[0, 1]
    facilities_yearly['eu_shortfall_tco2'].clip(lower=-5e6, upper=5e6).hist(bins=40, ax=ax, edgecolor='black', alpha=0.7, color='coral')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Shortfall (tCO₂)')
    ax.set_ylabel('Count')
    ax.set_title('Shortfall (verified - allocatedTotal)')

    # Allocation ratio (allocatedTotal / verified)
    ax = axes[1, 0]
    facilities_yearly['eu_alloc_ratio'].clip(lower=0, upper=3).hist(bins=40, ax=ax, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(1, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Allocation Ratio (allocTotal / verified)')
    ax.set_ylabel('Count')
    ax.set_title('Total Allocation Ratio (<1 = under-allocated)')

    plt.tight_layout()
    plt.show()

    # Time series for sample plants
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

    # Aggregate trends
    print("\n3. Aggregate Trends:")
    yearly_agg = facilities_yearly.groupby('year').agg({
        'eu_verified_tco2': 'sum',
        'eu_alloc_total_tco2': 'sum',
        fac_id_col: 'nunique'
    }).rename(columns={fac_id_col: 'n_plants'})  # type: ignore

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(yearly_agg.index, yearly_agg['eu_verified_tco2'] / 1e9, alpha=0.7, label='Verified', color='steelblue')
    ax1.plot(yearly_agg.index, yearly_agg['eu_alloc_total_tco2'] / 1e9, 'r-o', linewidth=2, label='Alloc Total')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('GtCO₂')
    ax1.set_title('Aggregate: Verified Emissions vs Total Allocated')
    ax1.legend(loc='upper right')
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # type: ignore
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(yearly_agg.index, yearly_agg['n_plants'], 'k--', linewidth=1, alpha=0.5)
    ax2.set_ylabel('# Plants', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    plt.tight_layout()
    plt.show()
