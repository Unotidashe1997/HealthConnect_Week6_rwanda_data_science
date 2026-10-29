# Hand-off: Data Science → ML Engineering (Week 7)

**Component:** admission-risk scorer (`src/admission_risk_scorer.py`, saved model `outputs/models/admission_scorer_v2.joblib`).
**Status:** model has **no demonstrated predictive skill** (hold-out AUC ≈ 0.53, CI includes 0.50). Use as an integration fixture only; show no risk percentages to end users.

## What to read
- `outputs/model_card.md` – purpose, limits, caution.
- `outputs/input_output_contract.json` – 18 input fields, allowed values and ranges; output columns.

## What to run (2 minutes)
1. Run Week 6 script 06 once so `outputs/data/HC_Integrated_Encounters_5000.csv` exists (the sample inputs).
2. `python week7/src/run_interface_tests.py` – prints PASS/FAIL for 10 checks (the full 18-case suite is in the notebook, section F).
3. Try your own pipeline's preprocessing output against the contract and note any mismatch (column names, types, category spellings, missing-value handling).

## Please send back (to complete the cross-track record)
| Item | Your answer |
|---|---|
| Name / date of your run | |
| Result of `run_interface_tests.py` (paste output) | |
| Input format your pipeline produces (differences from the contract) | |
| Any failure or mismatch found | |
| Change made / agreed | |
| Retest result | |

Known pitfalls: read CSVs with `keep_default_na=False` (the text `None` in `ChronicCondition` is a real value); `PatientVisits` must be "visits before this appointment" in production.
