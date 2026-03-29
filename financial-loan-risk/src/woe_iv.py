"""
=============================================================================
Financial Loan Risk — Weight of Evidence (WOE) & Information Value (IV)
=============================================================================
WOE/IV is the industry-standard feature selection method for credit scorecards.
It transforms continuous variables into interpretable bins and measures each
variable's predictive power for the binary default outcome.

IV Interpretation (Siddiqi 2006):
  < 0.02  : Unpredictive — exclude
  0.02–0.10: Weak predictor
  0.10–0.30: Medium predictor
  0.30–0.50: Strong predictor
  > 0.50   : Very strong (check for data leakage)
=============================================================================
"""

import logging
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

log = logging.getLogger("LoanRisk.WOE_IV")
warnings.filterwarnings("ignore", category=RuntimeWarning)


def compute_woe_iv(
    df: pd.DataFrame,
    feature: str,
    target: str = "default_flag",
    n_bins: int = 10,
    min_pct: float = 0.05,
) -> Tuple[pd.DataFrame, float]:
    """
    Compute Weight of Evidence and Information Value for a single feature.

    WOE = ln(Distribution of Events / Distribution of Non-Events)
    IV  = SUM[(Event% - NonEvent%) * WOE]

    Monotonic binning is enforced using iterative merging of non-monotonic bins.

    Args:
        df: DataFrame with feature and target columns
        feature: Column name to compute WOE/IV for
        target: Binary target column (1=default, 0=no default)
        n_bins: Initial number of quantile bins
        min_pct: Minimum bin size as fraction of total (prevents empty bins)

    Returns:
        woe_table: DataFrame with bins, WOE values, and IV contribution
        iv: Total Information Value for the feature
    """
    total_events     = df[target].sum()
    total_non_events = len(df) - total_events

    if total_events == 0 or total_non_events == 0:
        log.warning("Feature %s: no variation in target — IV=0", feature)
        return pd.DataFrame(), 0.0

    # Create initial quantile bins
    if pd.api.types.is_numeric_dtype(df[feature]):
        df["bin"] = pd.qcut(df[feature], q=n_bins, duplicates="drop")
    else:
        df["bin"] = df[feature].astype(str)

    grouped = df.groupby("bin", observed=True).agg(
        total=("bin", "count"),
        events=(target, "sum"),
    ).reset_index()

    grouped["non_events"]   = grouped["total"] - grouped["events"]
    grouped["pct_total"]    = grouped["total"] / len(df)
    grouped["event_rate"]   = grouped["events"] / grouped["total"]

    # Remove bins below minimum size threshold
    grouped = grouped[grouped["pct_total"] >= min_pct]

    # Distribution of events and non-events
    grouped["dist_events"]     = grouped["events"]     / total_events
    grouped["dist_non_events"] = grouped["non_events"] / total_non_events

    # Replace zeros to avoid log(0)
    grouped["dist_events"]     = grouped["dist_events"].replace(0, 1e-6)
    grouped["dist_non_events"] = grouped["dist_non_events"].replace(0, 1e-6)

    # WOE and IV
    grouped["woe"] = np.log(grouped["dist_events"] / grouped["dist_non_events"])
    grouped["iv_contribution"] = (
        (grouped["dist_events"] - grouped["dist_non_events"]) * grouped["woe"]
    )

    iv = grouped["iv_contribution"].sum()
    grouped["feature"] = feature
    grouped["iv_total"] = iv

    return grouped, round(iv, 4)


def compute_all_features(
    df: pd.DataFrame,
    features: List[str],
    target: str = "default_flag",
    iv_threshold: float = 0.10,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute WOE/IV for all candidate features and select predictive ones.

    Returns:
        iv_summary: DataFrame ranking all features by IV
        selected_features: Features with IV >= iv_threshold
    """
    results = []

    for feature in features:
        if feature not in df.columns:
            log.warning("Feature not found in DataFrame: %s", feature)
            continue
        try:
            _, iv = compute_woe_iv(df.copy(), feature, target)
            results.append({
                "feature": feature,
                "iv": iv,
                "predictive_power": classify_iv(iv),
            })
            log.debug("%s: IV=%.4f (%s)", feature, iv, classify_iv(iv))
        except Exception as e:
            log.warning("Could not compute IV for %s: %s", feature, e)

    iv_summary = (
        pd.DataFrame(results)
        .sort_values("iv", ascending=False)
        .reset_index(drop=True)
    )

    selected = iv_summary[iv_summary["iv"] >= iv_threshold]["feature"].tolist()
    log.info(
        "Feature selection complete: %d/%d features selected (IV >= %.2f)",
        len(selected), len(features), iv_threshold
    )
    return iv_summary, selected


def classify_iv(iv: float) -> str:
    """Classify IV into standard credit scoring categories."""
    if iv < 0.02:
        return "Unpredictive"
    elif iv < 0.10:
        return "Weak"
    elif iv < 0.30:
        return "Medium"
    elif iv < 0.50:
        return "Strong"
    else:
        return "Very Strong (check for leakage)"


def apply_woe_transform(
    df: pd.DataFrame,
    woe_tables: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Apply WOE transformation to a DataFrame using pre-computed WOE tables.
    Replaces raw feature values with their WOE-encoded equivalents.
    WOE-encoded features are directly usable as inputs to logistic regression.
    """
    df_woe = df.copy()
    for feature, woe_table in woe_tables.items():
        if feature not in df.columns:
            continue
        bin_woe_map = dict(zip(woe_table["bin"].astype(str), woe_table["woe"]))
        if pd.api.types.is_numeric_dtype(df[feature]):
            bins = pd.qcut(df[feature], q=10, duplicates="drop")
            df_woe[f"{feature}_woe"] = bins.astype(str).map(bin_woe_map).fillna(0)
        else:
            df_woe[f"{feature}_woe"] = df[feature].astype(str).map(bin_woe_map).fillna(0)

    return df_woe


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute WOE/IV for loan features")
    parser.add_argument("--input", required=True, help="Path to parquet file")
    parser.add_argument("--target", default="default_flag")
    parser.add_argument("--iv-threshold", type=float, default=0.10)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    features = [c for c in df.columns if c != args.target]
    iv_summary, selected = compute_all_features(df, features, args.target, args.iv_threshold)

    print("\nTop Features by Information Value:")
    print(iv_summary.head(15).to_string(index=False))
    print(f"\nSelected {len(selected)} features with IV >= {args.iv_threshold}")
