"""
Diagnostics and Visualization
==============================

Functions for validating data, checking covariate balance,
and visualizing treatment distributions and results.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure # type: ignore
from continuous import run_outcome_models

# Default values (can be overridden via function parameters)
FAC_ID_COL = "idx"
YEAR_COL = "year"
TREATMENT_COL = "eu_alloc_ratio"
TREATMENT_DISCRETE_COL = "treated"
OUTCOME_COL = "log_ets_co2"
CLUSTER_COL = "nuts2_region"
COHORT_COL = "cohort"
FIGURE_SIZE = (10, 6)
FIGURE_DPI = 100
COLOR_PALETTE = "colorblind"
LOG_ETS_CO2_COL = "log_ets_co2"
CONTROLS = ["capacity_mw", "share_coal", "share_gas", "share_oil", "share_other_gas"]


# =============================================================================
# 1. Panel Diagnostics
# =============================================================================

def check_panel_balance(
    df: pd.DataFrame,
    id_col: str = FAC_ID_COL,
    time_col: str = YEAR_COL
) -> Dict:
    """
    Check panel balance and report statistics.
    
    Returns:
        Dict with balance metrics and diagnostics
    """
    # Observations per unit
    obs_per_unit = df.groupby(id_col)[time_col].count()
    
    # Unique time periods
    all_periods = sorted(df[time_col].unique())
    n_periods = len(all_periods)
    
    # Balance check
    n_balanced = (obs_per_unit == n_periods).sum()
    n_units = len(obs_per_unit)
    
    result = {
        "n_units": n_units,
        "n_periods": n_periods,
        "n_observations": len(df),
        "periods": all_periods,
        "n_balanced_units": n_balanced,
        "pct_balanced": 100 * n_balanced / n_units,
        "obs_per_unit_stats": {
            "mean": obs_per_unit.mean(),
            "min": obs_per_unit.min(),
            "max": obs_per_unit.max(),
            "std": obs_per_unit.std()
        }
    }
    
    print(f"Panel balance check:")
    print(f"  Units: {n_units:,}")
    print(f"  Periods: {n_periods} ({all_periods[0]}-{all_periods[-1]})")
    print(f"  Observations: {len(df):,}")
    print(f"  Balanced units: {n_balanced:,} ({result['pct_balanced']:.1f}%)")
    print(f"  Obs/unit: mean={obs_per_unit.mean():.1f}, "
          f"min={obs_per_unit.min()}, max={obs_per_unit.max()}")
    
    return result


def check_missing_data(
    df: pd.DataFrame,
    key_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Report missing data patterns for key columns.
    """
    if key_cols is None:
        key_cols = [TREATMENT_COL, OUTCOME_COL, "capacity_mw", 
                    "share_coal", "share_gas", "eu_verified_tco2"]
    
    key_cols = [c for c in key_cols if c in df.columns]
    
    missing = df[key_cols].isnull().sum()
    pct_missing = 100 * missing / len(df)
    
    report = pd.DataFrame({
        "missing": missing,
        "pct_missing": pct_missing.round(2),
        "n_valid": len(df) - missing
    })
    
    print("Missing data report:")
    print(report.to_string())
    
    return report


# =============================================================================
# 2. Treatment Distribution
# =============================================================================

def plot_treatment_distribution(
    df: pd.DataFrame,
    treatment_col: str = TREATMENT_COL,
    by_year: bool = True,
    figsize: Tuple[int, int] = (12, 5)
) -> Figure:
    """
    Plot distribution of continuous treatment variable.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=FIGURE_DPI)
    
    # Overall distribution
    ax = axes[0]
    df[treatment_col].hist(bins=50, ax=ax, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(1.0, color="red", linestyle="--", linewidth=2, label="Threshold (ratio=1)")
    ax.set_xlabel(f"{treatment_col}", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Distribution of Allocation Ratio", fontsize=12)
    ax.legend()
    
    # By year (if requested)
    if by_year and YEAR_COL in df.columns:
        ax = axes[1]
        for year in sorted(df[YEAR_COL].unique()):
            subset = df[df[YEAR_COL] == year][treatment_col]
            ax.hist(subset, bins=30, alpha=0.4, label=str(year))
        ax.axvline(1.0, color="red", linestyle="--", linewidth=2)
        ax.set_xlabel(f"{treatment_col}", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title("Allocation Ratio by Year", fontsize=12)
        ax.legend(title="Year", fontsize=9)
    else:
        # Treatment over time
        ax = axes[1]
        yearly_mean = df.groupby(YEAR_COL)[treatment_col].mean()
        yearly_median = df.groupby(YEAR_COL)[treatment_col].median()
        ax.plot(yearly_mean.index, yearly_mean.values, "o-", label="Mean", color="steelblue")
        ax.plot(yearly_median.index, yearly_median.values, "s--", label="Median", color="orange")
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel(f"Mean {treatment_col}", fontsize=11)
        ax.set_title("Treatment Intensity Over Time", fontsize=12)
        ax.legend()
    
    plt.tight_layout()
    return fig


def plot_treatment_timing(
    df: pd.DataFrame,
    cohort_col: str = COHORT_COL,
    id_col: str = FAC_ID_COL,
    figsize: Tuple[int, int] = (10, 6)
) -> Figure:
    """
    Visualize treatment timing (cohort distribution).
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    
    # Cohort counts
    cohort_counts = df.groupby(id_col)[cohort_col].first().value_counts().sort_index()
    
    # Separate never-treated
    never_treated = cohort_counts.get(0, 0)
    treated_cohorts = cohort_counts[cohort_counts.index > 0]
    
    # Plot
    x = list(str(treated_cohorts.index)) + ["Never\nTreated"] # type: ignore
    y = list(treated_cohorts.values) + [never_treated] # type: ignore
    colors = ["steelblue"] * len(treated_cohorts) + ["gray"]
    
    bars = ax.bar(range(len(x)), y, color=colors, edgecolor="white", alpha=0.8)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation=0)
    ax.set_xlabel("Treatment Cohort (First Year Treated)", fontsize=11)
    ax.set_ylabel("Number of Facilities", fontsize=11)
    ax.set_title("Distribution of Treatment Timing", fontsize=12, fontweight="bold")
    
    # Add count labels
    for bar, count in zip(bars, y):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(int(count)), ha="center", va="bottom", fontsize=10)
    
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig

# =============================================================================
# 3. Outcome Visualization
# =============================================================================

def plot_outcome_trends(
    df: pd.DataFrame,
    outcome_col: str = OUTCOME_COL,
    treatment_col: str = TREATMENT_DISCRETE_COL,
    figsize: Tuple[int, int] = (10, 6)
) -> Figure:
    """
    Plot outcome trends for treated vs control groups.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    
    # Get ever-treated indicator
    ever_treated = df.groupby(FAC_ID_COL)[treatment_col].max()
    df = df.merge(ever_treated.rename("ever_treated"), on=FAC_ID_COL, how="left") # type: ignore
    
    # Mean outcome by year and treatment status
    trends = df.groupby([YEAR_COL, "ever_treated"])[outcome_col].mean().unstack()
    
    if 0 in trends.columns:
        ax.plot(trends.index, trends[0], "o-", label="Never Treated", 
                color="steelblue", linewidth=2, markersize=8)
    if 1 in trends.columns:
        ax.plot(trends.index, trends[1], "s-", label="Ever Treated", 
                color="coral", linewidth=2, markersize=8)
    
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel(f"Mean {outcome_col}", fontsize=11)
    ax.set_title("Outcome Trends: Treated vs Never Treated", fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_treatment_outcome_scatter(
    df: pd.DataFrame,
    treatment_col: str = TREATMENT_COL,
    outcome_col: str = OUTCOME_COL,
    color_by: Optional[str] = YEAR_COL,
    figsize: Tuple[int, int] = (10, 6)
) -> Figure:
    """
    Scatter plot of treatment vs outcome.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    
    df_plot = df.dropna(subset=[treatment_col, outcome_col])
    
    if color_by and color_by in df_plot.columns:
        scatter = ax.scatter(
            df_plot[treatment_col], 
            df_plot[outcome_col],
            c=df_plot[color_by],
            cmap="viridis",
            alpha=0.5,
            s=30
        )
        plt.colorbar(scatter, ax=ax, label=color_by)
    else:
        ax.scatter(
            df_plot[treatment_col], 
            df_plot[outcome_col],
            alpha=0.5,
            s=30,
            color="steelblue"
        )
    
    # Add regression line
    from scipy import stats
    slope, intercept, r, p, se = stats.linregress(
        df_plot[treatment_col], df_plot[outcome_col]
    )
    x_line = np.linspace(df_plot[treatment_col].min(), df_plot[treatment_col].max(), 100)
    ax.plot(x_line, intercept + slope * x_line, "r--", # type: ignore
            label=f"OLS: slope={slope:.4f} (p={p:.4f})")
    
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.5, label="Ratio = 1")
    
    ax.set_xlabel(f"{treatment_col}", fontsize=11)
    ax.set_ylabel(f"{outcome_col}", fontsize=11)
    ax.set_title("Treatment vs Outcome", fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig




# =============================================================================
# 5. Results Summary
# =============================================================================

def summarize_estimation_results(
    results: Dict,
    specification_name: str = "Main"
) -> None:
    """
    Print formatted summary of estimation results.
    """
    print(f"\n{'='*60}")
    print(f"Estimation Results: {specification_name}")
    print(f"{'='*60}")
    
    coef = results.get("coef", np.nan)
    se = results.get("se", np.nan)
    pval = results.get("pvalue", np.nan)
    ci = results.get("ci", (np.nan, np.nan))
    
    # Significance stars
    stars = ""
    if pval < 0.01:
        stars = "***"
    elif pval < 0.05:
        stars = "**"
    elif pval < 0.1:
        stars = "*"
    
    print(f"Coefficient: {coef:.6f}{stars}")
    print(f"Std. Error:  {se:.6f}")
    print(f"P-value:     {pval:.4f}")
    print(f"95% CI:      [{ci[0]:.6f}, {ci[1]:.6f}]")
    print(f"N obs:       {results.get('n_obs', 'N/A'):,}")
    print(f"N units:     {results.get('n_entities', 'N/A'):,}")
    
    if "formula" in results:
        print(f"Formula:     {results['formula']}")
    if "cluster_col" in results:
        print(f"Clustering:  {results['cluster_col']}")
    
    print(f"{'='*60}\n")


# =============================================================================
# 6. Heterogeneity Analysis
# =============================================================================

def run_heterogeneity_analysis(
    panel: pd.DataFrame,
    outcome_col: str,
    treatment_col: str = TREATMENT_COL,
    controls: List[str] = [], 
    cluster_col: str = CLUSTER_COL,
    id_col: str = FAC_ID_COL,
    electricity_col: str = "is_electricity",
    urban_col: str = "in_urban_area",
    min_obs: int = 20,
    min_facilities: int = 5
) -> pd.DataFrame:
    """
    Run heterogeneity analysis across multiple dimensions for a single outcome.
    
    Parameters
    ----------
    panel : DataFrame
        Panel data with outcome, treatment, and split variables
    outcome_col : str
        Outcome variable column name
    treatment_col : str
        Treatment variable column name
    controls : List[str]
        Control variables (include embeddings if applicable)
    cluster_col : str
        Clustering column for SEs
    electricity_col : str
        Column for electricity sector split
    urban_col : str
        Column for urban/rural split
    min_obs : int
        Minimum observations required per subsample
    min_facilities : int
        Minimum facilities required per subsample
        
    Returns
    -------
    DataFrame with heterogeneity results
    """
    if controls is None:
        controls = CONTROLS
    
    rows = []
    
    def add_row(dimension, group, res):
        if res and "coefficient" in res:
            pval = res["pvalue"]
            rows.append({
                "Dimension": dimension,
                "Group": group,
                "Coef": res["coefficient"],
                "SE": res["se"],
                "P-value": pval,
                "N": res.get("n_obs", ""),
                "Sig": "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            })
    
    def run_split(dim_name, col, val_true, val_false, label_true, label_false):
        for val, label in [(val_true, label_true), (val_false, label_false)]:
            if col not in panel.columns:
                continue
            subset = panel[panel[col] == val]
            if len(subset) >= min_obs and subset[id_col].nunique() >= min_facilities: # type: ignore
                try:
                    res = run_outcome_models(
                        subset, outcome_col=outcome_col, # type: ignore
                        treatment_col=treatment_col, controls=controls, cluster_col=cluster_col
                    )
                    add_row(dim_name, label, res)
                except:
                    pass
    
    # 1. Electricity sector
    run_split("Sector", electricity_col, True, False, "Electricity", "Other Sectors")
    
    # 2. Urbanization
    run_split("Location", urban_col, True, False, "Urban", "Rural")
    
    # 3. Fuel type - by dominant fuel
    fuel_cols = ["share_coal", "share_gas", "share_oil", "share_biomass"]
    fuel_cols = [c for c in fuel_cols if c in panel.columns]
    if fuel_cols:
        panel = panel.copy()
        # Determine dominant fuel for each observation
        fuel_shares = panel[fuel_cols].fillna(0)
        panel["_dominant_fuel"] = fuel_shares.idxmax(axis=1).str.replace("share_", "").str.title() # type: ignore
        
        # Run for each dominant fuel type with enough observations
        for fuel in panel["_dominant_fuel"].unique():
            subset = panel[panel["_dominant_fuel"] == fuel]
            if len(subset) >= min_obs and subset[id_col].nunique() >= min_facilities: # type: ignore
                try:
                    res = run_outcome_models(
                        subset, outcome_col=outcome_col, # type: ignore
                        treatment_col=treatment_col, controls=controls, cluster_col=cluster_col
                    )
                    add_row("Fuel", fuel, res)
                except:
                    pass
    
    # 4. Top countries
    if "country_code" in panel.columns:
        top_countries = panel.groupby("country_code")[id_col].nunique().nlargest(5).index # type: ignore
        for country in top_countries:
            subset = panel[panel["country_code"] == country]
            if len(subset) >= min_obs and subset[id_col].nunique() >= min_facilities: # type: ignore
                try:
                    res = run_outcome_models(
                        subset, outcome_col=outcome_col, # type: ignore
                        treatment_col=treatment_col, controls=controls, cluster_col=cluster_col
                    )
                    add_row("Country", str(country), res)
                except:
                    pass
    
    # 6. Top PyPSA clusters (electricity grid regions - electricity only)
    if "pypsa_cluster" in panel.columns and electricity_col in panel.columns:
        elec_panel = panel[panel[electricity_col] == True]
        if len(elec_panel) >= min_obs:
            top_pypsa = elec_panel.groupby("pypsa_cluster")[id_col].nunique().nlargest(5).index # type: ignore
            for cluster in top_pypsa:
                subset = elec_panel[elec_panel["pypsa_cluster"] == cluster]
                if len(subset) >= min_obs and subset[id_col].nunique() >= min_facilities: # type: ignore
                    try:
                        res = run_outcome_models(
                            subset, outcome_col=outcome_col, # type: ignore
                            treatment_col=treatment_col, controls=controls, cluster_col=cluster_col
                        )
                        add_row("PyPSA (Elec)", str(cluster), res)
                    except:
                        pass
    
    return pd.DataFrame(rows)


def run_full_heterogeneity_analysis(
    panel: pd.DataFrame,
    ets_col: str = LOG_ETS_CO2_COL,
    nox_col: str = "beirle_nox_kg_s",
    treatment_col: str = TREATMENT_COL,
    base_controls: List[str] = [],
    cluster_col: str = CLUSTER_COL,
    id_col: str = FAC_ID_COL,
    electricity_col: str = "is_electricity",
    urban_col: str = "in_urban_area"
) -> Dict[str, pd.DataFrame]:
    """
    Run heterogeneity analysis for all 5 TWFE specifications.
    
    Returns dict mapping spec name -> results DataFrame.
    """
    from embedding_reduction import reduce_embeddings, get_reduced_embedding_cols
    
    if base_controls is []:
        base_controls = ["capacity_mw", "share_coal", "share_gas"]
    
    results = {}
    
    # 1. ETS CO₂
    print("Running heterogeneity: ETS CO₂...")
    results["ETS CO₂"] = run_heterogeneity_analysis(
        panel, outcome_col=ets_col, treatment_col=treatment_col,
        controls=base_controls, cluster_col=cluster_col, id_col=id_col,
        electricity_col=electricity_col, urban_col=urban_col
    )
    
    # 2-5. NOx: (PCA/PLS) × (Permissive/Conservative)
    if nox_col not in panel.columns:
        return results
    
    for dl_col, dl_label in [("above_dl_0_03", "DL≥0.03"), ("above_dl_0_11", "DL≥0.11")]:
        if dl_col not in panel.columns:
            continue
        nox_sample = panel[panel[dl_col] == True].copy()
        if len(nox_sample) < 50:
            print(f"  Skipping {dl_label}: insufficient data")
            continue
        
        for emb_method in ["pca", "pls"]:
            spec_name = f"NOx ({emb_method.upper()}, {dl_label})"
            print(f"Running heterogeneity: {spec_name}...")
            try:
                df_reduced = reduce_embeddings(
                    nox_sample.copy(), method=emb_method, n_components=10, # type: ignore
                    target_col=nox_col if emb_method == "pls" else None
                )
                emb_cols = get_reduced_embedding_cols(df_reduced, method=emb_method)
                controls_with_emb = base_controls + emb_cols
                
                results[spec_name] = run_heterogeneity_analysis(
                    df_reduced, outcome_col=nox_col, treatment_col=treatment_col,
                    controls=controls_with_emb, cluster_col=cluster_col, id_col=id_col,
                    electricity_col=electricity_col, urban_col=urban_col
                )
            except Exception as e:
                print(f"  Error: {e}")
    
    return results


def display_heterogeneity_results(results: Dict[str, pd.DataFrame]) -> None:
    """Display heterogeneity results as formatted tables."""
    print("\n" + "=" * 70)
    print("HETEROGENEITY RESULTS SUMMARY")
    print("=" * 70)
    
    for spec_name, df in results.items():
        if df is None or len(df) == 0:
            continue
        print(f"\n### {spec_name}")
        df_display = df.copy()
        df_display["Coef"] = df_display["Coef"].apply(lambda x: f"{x:.4f}")
        df_display["SE"] = df_display["SE"].apply(lambda x: f"{x:.4f}")
        df_display["P-value"] = df_display["P-value"].apply(lambda x: f"{x:.4f}")
        
        # Use pandas display
        from IPython.display import display
        display(df_display)
    
    print("\n" + "-" * 70)
    print("Note: Negative coef = policy stringency reduces emissions")
    print("      * p<0.1, ** p<0.05, *** p<0.01")


# =============================================================================
# 7. Continuous Interaction Analysis (Fuel, Capacity, Urbanization)
# =============================================================================

def run_continuous_interaction_analysis(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str = TREATMENT_COL,
    controls: List[str] = [],
    cluster_col: str = CLUSTER_COL,
    id_col: str = FAC_ID_COL,
    time_col: str = YEAR_COL,
    urban_col: str = "in_urban_area"
) -> pd.DataFrame:
    """
    Estimate treatment effect heterogeneity using interaction terms.
    
    Interactions included:
    - Treatment × fuel shares (coal, gas, oil, biomass)
    - Treatment × capacity_mw
    - Treatment × urban (binary)
    
    Interpretation:
    - Positive interaction = weaker (less negative) treatment effect for that characteristic
    
    Returns DataFrame with interaction coefficients.
    """
    import pyfixest as pf
    
    df = df.copy()
    df = df.dropna(subset=[outcome_col, treatment_col])
    
    interaction_cols = []
    main_effect_cols = []
    
    # 1. Fuel share interactions
    fuel_cols = ["share_coal", "share_gas", "share_oil", "share_biomass"]
    fuel_cols = [c for c in fuel_cols if c in df.columns]
    for fuel_col in fuel_cols:
        df[f"treat_x_{fuel_col}"] = df[treatment_col] * df[fuel_col].fillna(0)
        interaction_cols.append(f"treat_x_{fuel_col}")
        main_effect_cols.append(fuel_col)
    
    # 2. Capacity interaction (standardized for interpretability)
    if "capacity_mw" in df.columns:
        cap_mean = df["capacity_mw"].mean()
        cap_std = df["capacity_mw"].std()
        df["capacity_std"] = (df["capacity_mw"] - cap_mean) / cap_std
        df["treat_x_capacity"] = df[treatment_col] * df["capacity_std"]
        interaction_cols.append("treat_x_capacity")
        main_effect_cols.append("capacity_std")
    
    # 3. Urban interaction
    if urban_col in df.columns:
        df["_urban"] = df[urban_col].astype(float).fillna(0)
        df["treat_x_urban"] = df[treatment_col] * df["_urban"]
        interaction_cols.append("treat_x_urban")
        main_effect_cols.append("_urban")
    
    if not interaction_cols:
        print("No interaction variables found")
        return pd.DataFrame()
    
    # Build formula
    base_controls = [c for c in controls if c not in main_effect_cols and c != "capacity_mw"]
    all_vars = [treatment_col] + interaction_cols + base_controls + main_effect_cols
    var_str = " + ".join(all_vars)
    fe_str = f"{id_col} + {cluster_col}^{time_col}"
    formula = f"{outcome_col} ~ {var_str} | {fe_str}"
    
    print(f"\nContinuous Interaction Model:")
    print(f"  Formula: {formula}")
    
    try:
        model = pf.feols(formula, data=df, vcov={"CRV1": cluster_col})
        
        # Extract results for treatment and interactions
        rows = []
        for var in [treatment_col] + interaction_cols:
            coef = model.coef().get(var, np.nan) # type: ignore
            se = model.se().get(var, np.nan) # type: ignore
            pval = model.pvalue().get(var, np.nan) # type: ignore
            
            # Clean up labels
            label = var
            label = label.replace("treat_x_share_", "× Fuel: ")
            label = label.replace("treat_x_capacity", "× Capacity (std)")
            label = label.replace("treat_x_urban", "× Urban")
            label = label.replace(treatment_col, "Treatment (baseline)")
            
            sig = ""
            if pval is not None and not np.isnan(pval):
                sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            
            rows.append({
                "Variable": label,
                "Coef": coef,
                "SE": se,
                "P-value": pval,
                "Sig": sig
            })
        
        return pd.DataFrame(rows)
    
    except Exception as e:
        print(f"  Error: {e}")
        return pd.DataFrame()


# Alias for backward compatibility
def run_fuel_interaction_analysis(*args, **kwargs):
    """Deprecated: Use run_continuous_interaction_analysis instead."""
    return run_continuous_interaction_analysis(*args, **kwargs)
