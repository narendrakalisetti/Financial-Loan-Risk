-- =============================================================================
-- LendRight UK — IFRS 9 ECL Reporting SQL Views
-- =============================================================================
-- Engine: PostgreSQL / DuckDB / Azure Synapse Serverless
-- Author: Narendra Kalisetti
-- =============================================================================

-- 1. IFRS 9 Stage Allocation Summary
CREATE OR REPLACE VIEW v_ifrs9_stage_summary AS
SELECT
    ifrs9_stage,
    CASE ifrs9_stage
        WHEN 1 THEN 'Stage 1 — Performing (12-month ECL)'
        WHEN 2 THEN 'Stage 2 — Underperforming (Lifetime ECL)'
        WHEN 3 THEN 'Stage 3 — Credit-Impaired (Lifetime ECL)'
    END                                             AS stage_description,
    COUNT(*)                                        AS loan_count,
    ROUND(SUM(ead), 2)                              AS ead_total_gbp,
    ROUND(SUM(ecl), 2)                              AS ecl_total_gbp,
    ROUND(AVG(pd_adjusted) * 100, 2)               AS avg_pd_pct,
    ROUND(AVG(lgd) * 100, 2)                        AS avg_lgd_pct,
    ROUND(SUM(ecl) / NULLIF(SUM(ead), 0) * 100, 2) AS coverage_ratio_pct
FROM loan_ecl_results
GROUP BY ifrs9_stage
ORDER BY ifrs9_stage;

-- 2. NPL Monitoring View
CREATE OR REPLACE VIEW v_npl_monitoring AS
SELECT
    reporting_date,
    COUNT(*) FILTER (WHERE days_past_due >= 90)     AS npl_count,
    COUNT(*)                                         AS total_loans,
    ROUND(COUNT(*) FILTER (WHERE days_past_due >= 90) * 100.0
          / NULLIF(COUNT(*), 0), 2)                  AS npl_rate_pct,
    SUM(ead) FILTER (WHERE days_past_due >= 90)      AS npl_ead_gbp,
    SUM(ead)                                         AS total_ead_gbp,
    -- Early warning: 30-60 DPD bucket (Stage 2 pipeline)
    ROUND(COUNT(*) FILTER (WHERE days_past_due BETWEEN 30 AND 89) * 100.0
          / NULLIF(COUNT(*), 0), 2)                  AS stage2_rate_pct,
    -- Alert flag
    CASE WHEN COUNT(*) FILTER (WHERE days_past_due >= 90) * 100.0
              / NULLIF(COUNT(*), 0) > 3.0
         THEN 'ALERT' ELSE 'NORMAL'
    END                                              AS npl_status
FROM loan_portfolio
GROUP BY reporting_date
ORDER BY reporting_date DESC;

-- 3. Scorecard Monitoring (PSI)
CREATE OR REPLACE VIEW v_scorecard_monitoring AS
SELECT
    monitoring_date,
    AVG(scorecard_score)                             AS avg_score,
    STDDEV(scorecard_score)                          AS std_score,
    PERCENTILE_CONT(0.25) WITHIN GROUP
        (ORDER BY scorecard_score)                   AS p25_score,
    PERCENTILE_CONT(0.75) WITHIN GROUP
        (ORDER BY scorecard_score)                   AS p75_score,
    COUNT(*) FILTER (WHERE risk_grade = 'A')         AS grade_a_count,
    COUNT(*) FILTER (WHERE risk_grade = 'B')         AS grade_b_count,
    COUNT(*) FILTER (WHERE risk_grade = 'C')         AS grade_c_count,
    COUNT(*) FILTER (WHERE risk_grade = 'D')         AS grade_d_count,
    COUNT(*) FILTER (WHERE risk_grade = 'E')         AS grade_e_count
FROM scored_applications
GROUP BY monitoring_date
ORDER BY monitoring_date DESC;
