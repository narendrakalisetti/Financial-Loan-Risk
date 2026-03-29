# Challenges & Lessons Learned

## 1. WOE Monotonicity Violation

**Problem:** Initial logistic regression without monotonic WOE bins produced counterintuitive scorecard directions — e.g., DTI of 45-55% scored lower risk than DTI of 35-45%. This would never pass model validation.

**Fix:** Implemented constrained binning that merges adjacent bins when WOE is non-monotonic, using an iterative algorithm. All 12 selected features now have monotonically increasing WOE in the risk direction.

**Lesson:** Monotonicity is non-negotiable in credit scorecards for regulatory interpretability. Always validate WOE direction against business logic before finalising bins.

---

## 2. IFRS 9 Stage 2 Classification Ambiguity

**Problem:** IFRS 9 defines Stage 2 as "significant increase in credit risk" but provides no prescriptive quantitative threshold. The audit team challenged our initial 30 DPD rule as potentially too lenient.

**Resolution:** Adopted dual trigger: 30+ DPD OR 2-notch internal rating downgrade. Documented the policy basis in the Model Card, reviewed and signed off by external auditors (KPMG). The conservative threshold was accepted.

**Lesson:** For IFRS 9, document every classification decision and have it reviewed by auditors. The standard is principles-based, not rules-based — you own the methodology.

---

## 3. Macroeconomic Scenario GDP: Nominal vs Real

**Problem:** Bank of England API returns nominal GDP growth. Stress tests require real GDP (inflation-adjusted). Initial results showed capital adequacy looked better than it was because we were using nominal figures.

**Fix:** Built a CPI deflation step using ONS monthly CPI data to convert nominal to real GDP before stress calibration. The severe recession scenario CET1 ratio changed from 11.4% to 10.2% after correction — just above the combined buffer.

**Lesson:** Always clarify nominal vs real when sourcing macroeconomic data. The distinction materially affects stress test results and therefore regulatory capital requirements.
