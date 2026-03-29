"""
=============================================================================
Financial Loan Risk — Basel III Stress Testing
=============================================================================
Runs PRA-aligned stress scenarios to assess capital adequacy.
Tests CET1 ratio under base, mild recession, and severe recession scenarios.

Basel III Minimum: CET1 >= 8.0%
PRA Stress Buffer:          2.5%
Combined Minimum:          10.5%
=============================================================================
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict

log = logging.getLogger("LoanRisk.StressTest")

# LendRight UK capital position (Dec 2023)
CAPITAL_BASE = {
    "cet1_capital_gbpm": 62.2,    # £62.2M CET1 capital
    "rwa_gbpm":          420.0,   # £420M Risk-Weighted Assets
    "cet1_ratio_pct":    14.8,    # 14.8% CET1 ratio
}

# Macro scenarios (aligned with Bank of England stress test framework)
SCENARIOS = {
    "base": {
        "label": "Base Case",
        "gdp_shock_pct":       0.0,
        "unemployment_pct":    4.2,
        "house_price_shock":   0.0,
        "interest_rate_change": 0.0,
    },
    "mild_recession": {
        "label": "Mild Recession",
        "gdp_shock_pct":       -2.0,
        "unemployment_pct":    6.5,
        "house_price_shock":   -8.0,
        "interest_rate_change": 1.5,
    },
    "severe_recession": {
        "label": "Severe Recession (PRA Annual Cyclical Scenario)",
        "gdp_shock_pct":       -5.0,
        "unemployment_pct":    9.8,
        "house_price_shock":  -20.0,
        "interest_rate_change": 3.0,
    },
    "extreme_stress": {
        "label": "Extreme Stress (Tail Risk)",
        "gdp_shock_pct":       -8.0,
        "unemployment_pct":   12.4,
        "house_price_shock":  -35.0,
        "interest_rate_change": 5.0,
    },
}


@dataclass
class StressResult:
    scenario: str
    label: str
    gdp_shock: float
    unemployment: float
    stressed_npl_rate: float
    stressed_ecl_gbpm: float
    capital_depletion_gbpm: float
    stressed_cet1_gbpm: float
    stressed_rwa_gbpm: float
    stressed_cet1_ratio: float
    meets_minimum: bool
    meets_combined_buffer: bool


def compute_stressed_npl(base_npl: float, scenario: dict) -> float:
    """
    Estimate stressed NPL rate using a simplified macro-credit linkage model.
    Based on LendRight internal stress calibration (validated by external auditor).

    Empirical relationship: NPL change ~ 0.3 * GDP shock + 0.2 * unemployment change
    """
    gdp_impact   = abs(scenario["gdp_shock_pct"]) * 0.30
    unemp_impact = (scenario["unemployment_pct"] - SCENARIOS["base"]["unemployment_pct"]) * 0.20
    house_impact = max(0, abs(scenario["house_price_shock"])) * 0.05
    stressed_npl = base_npl + gdp_impact + unemp_impact + house_impact
    return round(stressed_npl, 2)


def run_stress_scenario(
    scenario_key: str,
    base_ecl_gbpm: float = 32.4,
    base_npl_rate: float = 2.3,
) -> StressResult:
    """
    Run a single stress scenario and compute capital adequacy.

    Args:
        scenario_key: One of 'base', 'mild_recession', 'severe_recession', 'extreme_stress'
        base_ecl_gbpm: Current ECL provision in £M
        base_npl_rate: Current NPL rate as percentage
    """
    scenario = SCENARIOS[scenario_key]

    # Stressed NPL rate
    stressed_npl = compute_stressed_npl(base_npl_rate, scenario)

    # Stressed ECL — additional provision needed above base
    ecl_multiplier = 1 + (stressed_npl - base_npl_rate) / base_npl_rate
    stressed_ecl = base_ecl_gbpm * ecl_multiplier

    # Capital depletion: additional ECL above current provision depletes CET1
    capital_depletion = max(0, stressed_ecl - base_ecl_gbpm) * 0.72  # 72% post-tax

    # Stressed CET1 and RWA
    stressed_cet1 = CAPITAL_BASE["cet1_capital_gbpm"] - capital_depletion
    # RWA increases with stressed asset quality
    rwa_uplift = 1 + (stressed_npl - base_npl_rate) / 100 * 2
    stressed_rwa = CAPITAL_BASE["rwa_gbpm"] * rwa_uplift

    stressed_ratio = (stressed_cet1 / stressed_rwa * 100)

    result = StressResult(
        scenario=scenario_key,
        label=scenario["label"],
        gdp_shock=scenario["gdp_shock_pct"],
        unemployment=scenario["unemployment_pct"],
        stressed_npl_rate=stressed_npl,
        stressed_ecl_gbpm=round(stressed_ecl, 1),
        capital_depletion_gbpm=round(capital_depletion, 1),
        stressed_cet1_gbpm=round(stressed_cet1, 1),
        stressed_rwa_gbpm=round(stressed_rwa, 1),
        stressed_cet1_ratio=round(stressed_ratio, 1),
        meets_minimum=stressed_ratio >= 8.0,
        meets_combined_buffer=stressed_ratio >= 10.5,
    )

    log.info(
        "%s | NPL: %.1f%% | ECL: £%.1fM | CET1: %.1f%% | Min OK: %s | Buffer OK: %s",
        scenario_key, stressed_npl, stressed_ecl,
        stressed_ratio, result.meets_minimum, result.meets_combined_buffer
    )
    return result


def run_all_scenarios(
    base_ecl_gbpm: float = 32.4,
    base_npl_rate: float = 2.3,
) -> pd.DataFrame:
    """Run all stress scenarios and return summary DataFrame."""
    results = []
    for key in SCENARIOS:
        r = run_stress_scenario(key, base_ecl_gbpm, base_npl_rate)
        results.append({
            "Scenario":        r.label,
            "GDP Shock":       f"{r.gdp_shock:.0f}%",
            "Unemployment":    f"{r.unemployment:.1f}%",
            "NPL Rate":        f"{r.stressed_npl_rate:.1f}%",
            "ECL (£M)":        f"£{r.stressed_ecl_gbpm:.1f}M",
            "CET1 Ratio":      f"{r.stressed_cet1_ratio:.1f}%",
            "Meets Min (8%)":  "YES" if r.meets_minimum else "NO ❌",
            "Meets Buffer (10.5%)": "YES" if r.meets_combined_buffer else "BREACH ❌",
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="all",
                        choices=list(SCENARIOS.keys()) + ["all"])
    args = parser.parse_args()

    if args.scenario == "all":
        summary = run_all_scenarios()
        print("\nBasel III Stress Test Results:")
        print(summary.to_string(index=False))
    else:
        r = run_stress_scenario(args.scenario)
        print(f"\n{r.label}")
        print(f"  Stressed CET1 Ratio : {r.stressed_cet1_ratio}%")
        print(f"  Meets 8% minimum    : {'YES' if r.meets_minimum else 'NO'}")
        print(f"  Meets 10.5% buffer  : {'YES' if r.meets_combined_buffer else 'BREACH'}")
