"""
Unit tests for Financial Loan Risk analytics pipeline.
"""
import pytest
import numpy as np
import pandas as pd
from src.woe_iv import compute_woe_iv, classify_iv
from src.scorecard import compute_gini, compute_ks_statistic, scale_scores
from src.ecl_calculator import classify_stage, compute_discount_factor, compute_ecl
from src.stress_testing import run_stress_scenario, compute_stressed_npl


# ──────────────────────────────────────────────────────────────────
# WOE / IV Tests
# ──────────────────────────────────────────────────────────────────
class TestWoeIv:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 1000
        feature = np.random.normal(50, 15, n)
        # Correlated target: higher feature = lower default prob
        prob = 1 / (1 + np.exp((feature - 50) / 10))
        target = np.random.binomial(1, prob, n)
        return pd.DataFrame({"feature": feature, "default_flag": target})

    def test_iv_is_non_negative(self, sample_df):
        _, iv = compute_woe_iv(sample_df, "feature", "default_flag")
        assert iv >= 0

    def test_predictive_feature_has_positive_iv(self, sample_df):
        _, iv = compute_woe_iv(sample_df, "feature", "default_flag")
        assert iv > 0.02, f"Expected IV > 0.02, got {iv}"

    def test_classify_iv_correct(self):
        assert classify_iv(0.01)  == "Unpredictive"
        assert classify_iv(0.05)  == "Weak"
        assert classify_iv(0.15)  == "Medium"
        assert classify_iv(0.40)  == "Strong"
        assert "Very Strong" in classify_iv(0.60)


# ──────────────────────────────────────────────────────────────────
# Scorecard Tests
# ──────────────────────────────────────────────────────────────────
class TestScorecard:
    def test_gini_perfect_model(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        assert compute_gini(y_true, y_pred) == pytest.approx(1.0, abs=0.01)

    def test_gini_random_model(self):
        np.random.seed(0)
        y_true = np.random.randint(0, 2, 1000)
        y_pred = np.random.uniform(0, 1, 1000)
        # Random model should have Gini near 0
        assert abs(compute_gini(y_true, y_pred)) < 0.1

    def test_ks_range(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        ks = compute_ks_statistic(y_true, y_pred)
        assert 0 <= ks <= 1

    def test_score_scaling_range(self):
        log_odds = np.array([-3.0, 0.0, 3.0])
        scores = scale_scores(log_odds)
        assert scores.min() >= 300
        assert scores.max() <= 850

    def test_higher_pd_lower_score(self):
        """Higher probability of default should produce lower credit score."""
        low_risk_log_odds  = np.array([2.0])   # Low PD
        high_risk_log_odds = np.array([-2.0])  # High PD
        low_score  = scale_scores(low_risk_log_odds)[0]
        high_score = scale_scores(high_risk_log_odds)[0]
        assert low_score > high_score


# ──────────────────────────────────────────────────────────────────
# IFRS 9 ECL Tests
# ──────────────────────────────────────────────────────────────────
class TestECL:
    @pytest.fixture
    def sample_loan_df(self):
        return pd.DataFrame({
            "loan_id":            ["L001", "L002", "L003", "L004"],
            "days_past_due":      [0, 35, 95, 0],
            "rating_downgrades":  [0, 0, 0, 3],
            "legal_action":       [False, False, True, False],
            "outstanding_balance":[10000, 5000, 8000, 15000],
            "undrawn_commitment": [0, 0, 0, 0],
            "remaining_term_months": [36, 24, 12, 48],
            "product_type":       ["personal_loan", "credit_card", "personal_loan", "car_finance"],
            "pd_estimate":        [0.02, 0.08, 0.75, 0.03],
        })

    def test_stage_classification(self):
        assert classify_stage({"days_past_due": 0,  "rating_downgrades": 0, "legal_action": False}) == 1
        assert classify_stage({"days_past_due": 35, "rating_downgrades": 0, "legal_action": False}) == 2
        assert classify_stage({"days_past_due": 95, "rating_downgrades": 0, "legal_action": False}) == 3
        assert classify_stage({"days_past_due": 0,  "rating_downgrades": 3, "legal_action": False}) == 2
        assert classify_stage({"days_past_due": 0,  "rating_downgrades": 0, "legal_action": True})  == 3

    def test_stage_3_overrides_stage_2(self):
        """Stage 3 (90+ DPD) takes precedence over Stage 2 flags."""
        row = {"days_past_due": 95, "rating_downgrades": 3, "legal_action": True}
        assert classify_stage(row) == 3

    def test_discount_factor_range(self):
        assert compute_discount_factor(0)   == 1.0
        assert compute_discount_factor(12)  < 1.0
        assert compute_discount_factor(120) < compute_discount_factor(12)

    def test_ecl_non_negative(self, sample_loan_df):
        result = compute_ecl(sample_loan_df)
        assert (result["ecl"] >= 0).all()

    def test_stage3_higher_ecl_than_stage1(self, sample_loan_df):
        result = compute_ecl(sample_loan_df)
        s1_avg_ecl = result[result["ifrs9_stage"] == 1]["ecl"].mean()
        s3_avg_ecl = result[result["ifrs9_stage"] == 3]["ecl"].mean()
        assert s3_avg_ecl > s1_avg_ecl

    def test_severe_scenario_higher_ecl(self, sample_loan_df):
        base_result   = compute_ecl(sample_loan_df, scenario="base")
        severe_result = compute_ecl(sample_loan_df, scenario="severe")
        assert severe_result["ecl"].sum() >= base_result["ecl"].sum()


# ──────────────────────────────────────────────────────────────────
# Stress Testing Tests
# ──────────────────────────────────────────────────────────────────
class TestStressTesting:
    def test_base_scenario_unchanged_npl(self):
        result = run_stress_scenario("base", base_npl_rate=2.3)
        assert result.stressed_npl_rate == pytest.approx(2.3, abs=0.1)

    def test_severe_recession_lowers_cet1(self):
        base   = run_stress_scenario("base")
        severe = run_stress_scenario("severe_recession")
        assert severe.stressed_cet1_ratio < base.stressed_cet1_ratio

    def test_stressed_npl_increases_with_severity(self):
        base_npl    = compute_stressed_npl(2.3, {"gdp_shock_pct":  0.0, "unemployment_pct": 4.2, "house_price_shock": 0.0})
        severe_npl  = compute_stressed_npl(2.3, {"gdp_shock_pct": -5.0, "unemployment_pct": 9.8, "house_price_shock": -20.0})
        assert severe_npl > base_npl

    def test_cet1_ratio_is_positive(self):
        result = run_stress_scenario("severe_recession")
        assert result.stressed_cet1_ratio > 0

    def test_base_meets_all_buffers(self):
        result = run_stress_scenario("base")
        assert result.meets_minimum
        assert result.meets_combined_buffer
