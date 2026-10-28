# Model card — HealthConnect admission-risk scorer (week7-v2)
**Status: NOT approved for patient-level decisions.** Hold-out AUC 0.529 (95% CI 0.47–0.59); passes 0/6 pre-registered acceptance criteria.
- **Task:** predict hospital admission (11.7% of 5,000 encounters, 586 admissions) from booking-time information.
- **Model:** Candidate (Week 6), unchanged; scikit-learn 1.9.1; trained on all 5,000 encounters (Jan–Dec 2026).
- **Inputs:** see `input_output_contract.json` (18 fields; closed lists for gender, province, chronic condition; numeric ranges enforced).
- **Outputs:** risk_score, risk_band (higher/typical/lower = top/bottom 20% of training scores), warnings, model_version, use_caution.
- **Known limits:** no demonstrated skill; probabilities uninformative; PatientVisits counts the whole year (look-ahead) and must be recomputed as 'visits before this appointment' in production; trained on a single year from one data extract with inconsistent facility/department keys.
- **Intended use:** a *test fixture* for pipeline integration only. It is not a missed-appointment (no-show) model — the data contain no attendance field.
