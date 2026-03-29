"""
=============================================================================
Financial Loan Risk — IFRS 9 Expected Credit Loss (ECL) Calculator
=============================================================================
Implements IFRS 9 three-stage ECL model:
  Stage 1: 12-month ECL  (performing loans, no significant risk increase)
  Stage 2: Lifetime ECL  (significant increase in credit risk)
  Stage 3: Lifetime ECL  (credit-impaired / defaulted)

ECL Formula: ECL = PD x LGD x EAD x Discount Factor

Stage Classification Rules (LendRight UK policy, signed off by KPMG auditors):
  Stage 1 → Stage 2: 30+ DPD OR internal rating downgraded 2+ notches
  Stage 2 → Stage 3: 90+ DPD OR legal action commenced
=============================================================================
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

log = logging.getLogger("LoanRisk.ECL")
PROCESSED_DIR = Path("data/processed")

# Macroeconomic adjustment factor (Bank of England forward guidance)
# Applied to lifetime PD to reflect forecast economic conditions
MACRO_ADJUSTMENT = {
    "base":    1.00,
    "mild":    1.08,  # Mild recession scenario
    "severe":  1.22,  # Severe recession scenario
}

# Loss Given Default assumptions by product type (LendRight internal study)
LGD_BY_PRODUCT = {
    "personal_loan": 0.55,
    "car_finance":   0.35,  # Asset-backed — lower LGD
    "credit_card":   0.72,
    "default":       0.55,
}

# Effective interest rate for discounting (IFRS 9 original EIR method)
EFFECTIVE_INTEREST_RATE = 0.089  # 8.9% — LendRight portfolio average


def classify_stage(row: pd.Series) -> int:
    """
    Classify each loan into IFRS 9 stage based on LendRight policy.
    Stage 3 takes precedence over Stage 2.
    """
    # Stage 3: Credit-impaired
    if row.get("days_past_due", 0) >= 90 or row.get("legal_action", False):
        return 3
    # Stage 2: Significant increase in credit risk
    if row.get("days_past_due", 0) >= 30 or row.get("rating_downgrades", 0) >= 2:
        return 2
    # Stage 1: Performing
    return 1


def compute_discount_factor(
    remaining_months: int,
    effective_rate: float = EFFECTIVE_INTEREST_RATE
) -> float:
    """
    Discount future cash flows to present value using original EIR.
    IFRS 9 requires discounting ECL at the original effective interest rate.
    """
    monthly_rate = effective_rate / 12
    if remaining_months <= 0:
        return 1.0
    return 1 / (1 + monthly_rate) ** remaining_months


def compute_ecl(
    df: pd.DataFrame,
    pd_column: str = "pd_estimate",
    scenario: str = "base",
    as_of_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute IFRS 9 ECL for each loan in the portfolio.

    Args:
        df: Loan-level DataFrame with PD, product type, exposure, DPD columns
        pd_column: Column containing PD estimates from scorecard model
        scenario: Macroeconomic scenario ('base', 'mild', 'severe')
        as_of_date: Reporting date (defaults to today)

    Returns:
        DataFrame with ECL components and stage classification appended
    """
    df = df.copy()
    macro_factor = MACRO_ADJUSTMENT.get(scenario, 1.0)

    # Stage classification
    df["ifrs9_stage"] = df.apply(classify_stage, axis=1)

    # LGD by product type
    df["lgd"] = df.get("product_type", "default").map(
        lambda p: LGD_BY_PRODUCT.get(p, LGD_BY_PRODUCT["default"])
    )

    # EAD = Outstanding balance + 75% of undrawn commitment (credit conversion factor)
    df["ead"] = df["outstanding_balance"] + 0.75 * df.get("undrawn_commitment", 0)

    # Macro-adjusted PD
    df["pd_adjusted"] = (df[pd_column] * macro_factor).clip(0, 1)

    # Lifetime PD (simplified: 1 - (1-PD_12m)^remaining_years)
    df["remaining_years"] = df.get("remaining_term_months", 36) / 12
    df["pd_lifetime"] = (1 - (1 - df["pd_adjusted"]) ** df["remaining_years"]).clip(0, 1)

    # Discount factor
    df["discount_factor"] = df.get("remaining_term_months", 36).apply(
        lambda m: compute_discount_factor(int(m))
    )

    # ECL by stage
    df["ecl"] = np.where(
        df["ifrs9_stage"] == 1,
        # Stage 1: 12-month ECL
        df["pd_adjusted"] * df["lgd"] * df["ead"] * df["discount_factor"],
        np.where(
            df["ifrs9_stage"].isin([2, 3]),
            # Stage 2/3: Lifetime ECL
            df["pd_lifetime"] * df["lgd"] * df["ead"] * df["discount_factor"],
            0
        )
    )

    df["coverage_ratio"] = (df["ecl"] / df["ead"].replace(0, np.nan) * 100).round(2)

    log.info(
        "ECL computed | Stage 1: %d | Stage 2: %d | Stage 3: %d | Total ECL: £%.1fM | Scenario: %s",
        (df["ifrs9_stage"] == 1).sum(),
        (df["ifrs9_stage"] == 2).sum(),
        (df["ifrs9_stage"] == 3).sum(),
        df["ecl"].sum() / 1e6,
        scenario
    )
    return df


def ecl_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate IFRS 9 ECL summary table by stage."""
    summary = (
        df.groupby("ifrs9_stage")
        .agg(
            loan_count=("ecl", "count"),
            ead_total=("ead", "sum"),
            ecl_total=("ecl", "sum"),
            avg_pd=("pd_adjusted", "mean"),
            avg_lgd=("lgd", "mean"),
            avg_coverage=("coverage_ratio", "mean"),
        )
        .round({"avg_pd": 4, "avg_lgd": 4, "avg_coverage": 2})
        .reset_index()
    )
    summary["coverage_pct"] = (summary["ecl_total"] / summary["ead_total"] * 100).round(2)
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     required=True)
    parser.add_argument("--scenario",  default="base", choices=["base","mild","severe"])
    parser.add_argument("--as-of-date", default=None)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    result = compute_ecl(df, scenario=args.scenario)
    print("\nIFRS 9 ECL Summary:")
    print(ecl_summary(result).to_string(index=False))
    print(f"\nTotal ECL: £{result['ecl'].sum()/1e6:.2f}M")
    print(f"Portfolio Coverage: {result['ecl'].sum()/result['ead'].sum()*100:.2f}%")
