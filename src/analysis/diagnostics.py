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

# Default values (can be overridden via function parameters)
FAC_ID_COL = "idx"
YEAR_COL = "year"
TREATMENT_COL = "eu_alloc_ratio"
TREATMENT_DISCRETE_COL = "treated"
OUTCOME_COL = "log_ets_co2"
COHORT_COL = "cohort"
FIGURE_SIZE = (10, 6)
FIGURE_DPI = 100
COLOR_PALETTE = "colorblind"


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
# 3. Covariate Balance
# =============================================================================

def compute_covariate_balance(
    df: pd.DataFrame,
    treatment_col: str = TREATMENT_DISCRETE_COL,
    covariates: Optional[List[str]] = None,
    baseline_only: bool = True
) -> pd.DataFrame:
    """
    Compute covariate balance between treated and control groups.
    
    Reports standardized mean differences (SMD) and t-test p-values.
    
    Parameters:
        df: Panel DataFrame
        treatment_col: Binary treatment indicator
        covariates: List of covariate columns
        baseline_only: If True, use only pre-treatment observations
    
    Returns:
        DataFrame with balance statistics
    """
    if covariates is None:
        covariates = ["capacity_mw", "share_coal", "share_gas", "share_oil",
                      "share_biomass", "eu_verified_tco2"]
    
    covariates = [c for c in covariates if c in df.columns]
    
    if baseline_only and COHORT_COL in df.columns:
        # Use only pre-treatment observations
        df = df[(df[COHORT_COL] == 0) | (df[YEAR_COL] < df[COHORT_COL])] # type: ignore
    
    # Ever-treated indicator (facility level)
    ever_treated = df.groupby(FAC_ID_COL)[treatment_col].max()
    df = df.merge(ever_treated.rename("ever_treated"), on=FAC_ID_COL, how="left") # type: ignore
    
    # Compute means by group
    treated = df[df["ever_treated"] == 1]
    control = df[df["ever_treated"] == 0]
    
    results = []
    for cov in covariates:
        t_mean = treated[cov].mean()
        c_mean = control[cov].mean()
        t_std = treated[cov].std()
        c_std = control[cov].std()
        
        # Standardized mean difference
        pooled_std = np.sqrt((t_std**2 + c_std**2) / 2)
        smd = (t_mean - c_mean) / pooled_std if pooled_std > 0 else 0
        
        # T-test
        from scipy import stats
        t_stat, p_val = stats.ttest_ind(
            treated[cov].dropna(), # type: ignore
            control[cov].dropna(), # type: ignore
            equal_var=False
        )
        
        results.append({
            "covariate": cov,
            "treated_mean": t_mean,
            "control_mean": c_mean,
            "difference": t_mean - c_mean,
            "smd": smd,
            "abs_smd": abs(smd),
            "p_value": p_val
        })
    
    balance_df = pd.DataFrame(results)
    balance_df = balance_df.sort_values("abs_smd", ascending=False)
    
    # Summary
    n_imbalanced = (balance_df["abs_smd"] > 0.1).sum()
    print(f"Covariate balance check:")
    print(f"  Treated: {len(treated):,} obs, Control: {len(control):,} obs")
    print(f"  Covariates with |SMD| > 0.1: {n_imbalanced}/{len(covariates)}")
    
    return balance_df


def plot_covariate_balance(
    balance_df: pd.DataFrame,
    threshold: float = 0.1,
    figsize: Tuple[int, int] = (8, 6)
) -> Figure:
    """
    Love plot showing covariate balance (SMD).
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    
    # Sort by absolute SMD
    df = balance_df.sort_values("abs_smd", ascending=True)
    
    # Colors based on threshold
    colors = ["green" if abs(s) <= threshold else "red" for s in df["smd"]]
    
    # Plot
    y_pos = range(len(df))
    ax.barh(y_pos, df["smd"], color=colors, alpha=0.7, edgecolor="white")
    
    # Reference lines
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(-threshold, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(threshold, color="gray", linestyle="--", alpha=0.5)
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["covariate"])
    ax.set_xlabel("Standardized Mean Difference", fontsize=11)
    ax.set_title("Covariate Balance: Treated vs Control", fontsize=12, fontweight="bold")
    
    # Threshold annotation
    ax.annotate(f"|SMD| = {threshold}", xy=(threshold, len(df)-1), 
                xytext=(threshold + 0.05, len(df)-0.5), fontsize=9, alpha=0.7)
    
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


# =============================================================================
# 4. Outcome Visualization
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
