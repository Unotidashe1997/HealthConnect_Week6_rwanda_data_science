# Final model requirements — hand-off to ML Engineering (Week 8)

**Status:** Week 6 logistic regression (L2, C=0.1, 18 features), unchanged. Integration fixture only. **Not approved for patient-level decisions** (hold-out AUC 0.529, 0/6 criteria).

## Interface
- Use `week7/src/admission_risk_scorer.py` (`AdmissionScorerV2`, version `week7-v2`) and `week7/outputs/input_output_contract.json`.
- Inputs: 18 required fields; numeric ranges and category lists per the contract; invalid input raises `ScoringError` naming the field.
- Outputs per row: `risk_score`, `risk_band`, `warnings`, `model_version`, `use_caution` (always True at present).

## Data handling
- Read CSVs with `keep_default_na=False, na_values=[""]` (`ChronicCondition == "None"` is a real value).
- `PatientVisits` currently counts the whole year (look-ahead). Recompute as visits *before* the appointment.
- Never log or export names, phone numbers or emergency contacts.

## Do / do not
- Do: use for pipeline, API and logging tests; display the caution flag.
- Do not: show scores to staff or patients; use for individual decisions; use as a no-show predictor.

## Retraining plan
1. Obtain appointment status (attended / missed / cancelled).
2. Rebuild the target, recompute booking-time features.
3. Rerun the six criteria, the time-split test, calibration, and the planted-signal check (`week7/src/week7_model_testing.py`).
4. Deploy only if all criteria pass.

## Dependencies
Python 3.11+, pandas, numpy, scikit-learn 1.9.1, scipy, joblib (see `requirements.txt`). The joblib model must be loaded with the same scikit-learn version.
