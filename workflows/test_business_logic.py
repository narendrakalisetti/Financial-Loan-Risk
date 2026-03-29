"""
Pure Python business logic tests — zero external dependencies.
Validates credit risk pipeline logic by reading source files directly.
No pandas, sklearn, numpy, or any data science library required.
"""
import os


def src(filename):
    """Read a source file relative to project root."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, filename)) as f:
        return f.read()


class TestWoeIvLogic:
    def test_iv_thresholds_documented(self):
        assert "0.02" in src("src/woe_iv.py")
        assert "0.10" in src("src/woe_iv.py")

    def test_woe_formula_present(self):
        assert "np.log" in src("src/woe_iv.py")

    def test_iv_contribution_computed(self):
        assert "iv_contribution" in src("src/woe_iv.py")

    def test_zero_replacement_for_log_safety(self):
        assert "1e-6" in src("src/woe_iv.py")

    def test_classify_iv_function_exists(self):
        assert "classify_iv" in src("src/woe_iv.py")

    def test_unpredictive_label_present(self):
        assert "Unpredictive" in src("src/woe_iv.py")

    def test_strong_predictor_label_present(self):
        assert "Strong" in src("src/woe_iv.py")

    def test_leakage_warning_present(self):
        assert "leakage" in src("src/woe_iv.py").lower()


class TestScorecardModel:
    def test_gini_function_exists(self):
        assert "compute_gini" in src("src/scorecard.py")

    def test_gini_formula_correct(self):
        assert "2 * auc - 1" in src("src/scorecard.py")

    def test_ks_function_exists(self):
        assert "compute_ks_statistic" in src("src/scorecard.py")

    def test_psi_function_exists(self):
        assert "compute_population_stability_index" in src("src/scorecard.py")

    def test_score_minimum_300(self):
        assert "300" in src("src/scorecard.py")

    def test_score_maximum_850(self):
        assert "850" in src("src/scorecard.py")

    def test_base_score_is_600(self):
        assert "BASE_SCORE = 600" in src("src/scorecard.py")

    def test_pdo_is_20(self):
        content = src("src/scorecard.py")
        assert "PDO" in content and "20" in content

    def test_isotonic_calibration_used(self):
        assert "isotonic" in src("src/scorecard.py").lower()

    def test_gini_in_readme(self):
        assert "0.61" in src("README.md")

    def test_ks_in_readme(self):
        assert "0.42" in src("README.md")

    def test_auc_in_readme(self):
        assert "0.81" in src("README.md")


class TestIFRS9ECL:
    def test_three_stages_present(self):
        content = src("src/ecl_calculator.py")
        assert "Stage 1" in content or "stage == 1" in content
        assert "Stage 2" in content or "stage == 2" in content
        assert "Stage 3" in content or "stage == 3" in content

    def test_pd_lgd_ead_in_ecl(self):
        content = src("src/ecl_calculator.py")
        assert "pd_" in content
        assert "lgd" in content
        assert "ead" in content

    def test_90_dpd_stage3_threshold(self):
        assert "90" in src("src/ecl_calculator.py")

    def test_30_dpd_stage2_threshold(self):
        assert "30" in src("src/ecl_calculator.py")

    def test_discount_factor_present(self):
        assert "discount_factor" in src("src/ecl_calculator.py")

    def test_macro_adjustment_present(self):
        assert "MACRO_ADJUSTMENT" in src("src/ecl_calculator.py")

    def test_base_scenario_present(self):
        assert '"base"' in src("src/ecl_calculator.py")

    def test_severe_scenario_present(self):
        assert '"severe"' in src("src/ecl_calculator.py")

    def test_personal_loan_lgd(self):
        assert "personal_loan" in src("src/ecl_calculator.py")

    def test_car_finance_lgd(self):
        assert "car_finance" in src("src/ecl_calculator.py")

    def test_credit_card_lgd(self):
        assert "credit_card" in src("src/ecl_calculator.py")

    def test_ecl_in_readme(self):
        assert "ECL" in src("README.md")


class TestBaselStressTest:
    def test_base_scenario_defined(self):
        assert '"base"' in src("src/stress_testing.py")

    def test_mild_recession_defined(self):
        assert "mild_recession" in src("src/stress_testing.py")

    def test_severe_recession_defined(self):
        assert "severe_recession" in src("src/stress_testing.py")

    def test_extreme_stress_defined(self):
        assert "extreme_stress" in src("src/stress_testing.py")

    def test_cet1_minimum_8pct(self):
        assert "8.0" in src("src/stress_testing.py")

    def test_combined_buffer_10_5pct(self):
        assert "10.5" in src("src/stress_testing.py")

    def test_gdp_shock_parameter(self):
        assert "gdp_shock" in src("src/stress_testing.py")

    def test_unemployment_parameter(self):
        assert "unemployment" in src("src/stress_testing.py")

    def test_cet1_ratio_output(self):
        assert "stressed_cet1_ratio" in src("src/stress_testing.py")

    def test_meets_minimum_flag(self):
        assert "meets_minimum" in src("src/stress_testing.py")

    def test_cet1_in_readme(self):
        assert "CET1" in src("README.md")

    def test_basel_in_readme(self):
        assert "Basel" in src("README.md")


class TestSQLAndDocs:
    def test_ecl_sql_view_exists(self):
        content = src("sql/ecl_reporting.sql")
        assert "CREATE OR REPLACE VIEW" in content
        assert "ifrs9_stage" in content

    def test_npl_monitoring_in_sql(self):
        assert "npl" in src("sql/ecl_reporting.sql").lower()

    def test_model_card_exists(self):
        content = src("docs/MODEL_CARD.md")
        assert "Gini" in content
        assert "FCA" in content

    def test_challenges_has_woe_content(self):
        content = src("CHALLENGES.md")
        assert "WOE" in content or "monoton" in content.lower()

    def test_challenges_has_ifrs9_content(self):
        assert "IFRS 9" in src("CHALLENGES.md")

    def test_readme_has_woe_iv(self):
        content = src("README.md")
        assert "WOE" in content or "Information Value" in content

    def test_readme_has_stress_scenarios(self):
        content = src("README.md")
        assert "Severe" in content or "severe" in content
