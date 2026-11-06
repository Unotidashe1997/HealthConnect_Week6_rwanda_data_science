# HealthConnect Rwanda — Data Science Case Study (Weeks 6–8)

AnalystLab Africa · Experience Lab · Data Science track · **Project question:** *How can HealthConnect Clinic use data and AI to reduce missed appointments and improve the patient support experience?*

## Summary
Across three weeks I analysed 27 HealthConnect files (20,000 patients, 5,000 appointments, 60,000 bills), built an admission-risk model, tested it rigorously and documented the final decision. **The model has no usable predictive skill** (hold-out AUC 0.53, 95% CI 0.47–0.59, 0 of 6 pre-set criteria met), and the data contains no attendance field, so missed appointments cannot be modelled yet. The validated outputs are a transparent outreach rule, a flat visit forecast, collections findings and a reusable test harness.

| Week | Focus | Folder | Status |
|---|---|---|---|
| 6 | Data prep, EDA, statistics, forecasting, models, clustering | `src/`, `outputs/`, `notebooks/`, `docs/` | ✅ |
| 7 | Model testing, error analysis, refinement, ML hand-off | `week7/`, `week7_test/` | ✅ |
| 8 | Final model decision, documentation, cross-track package, presentation | `week8/` | ✅ |

## Final model decision (Week 8)
- Final model: Week 6 logistic regression, unchanged, kept as an **integration fixture**. Not approved for patient-level decisions.
- Cannot be used to rank patients, show risk percentages or predict no-shows.
- Can be used to test ML pipelines; the **rule-based list** (6,662 high-risk patients, 5,164 with no 2026 appointment) and the **forecast** (≈4,975 visits/month) are the usable outputs.
- Requirements for ML Engineering: `week8/FINAL_MODEL_REQUIREMENTS.md`.

## Key findings
- 0 of 26 admission factors significant after Holm correction; 0 of 15 hypothesis tests significant.
- 0.92 bn RWF billing unpaid; 21,028 claims (2.44 bn RWF) pending.
- Only 5,000 of 60,000 bills have an appointment record; keys are inconsistent.
- Planted-signal tests show the method detects true signals from AUC ≈ 0.57, so the "no signal" result reflects the data.

## Repository layout
```
src/            Week 6 pipeline (8 scripts)
notebooks/      Week 6 notebook
docs/           Week 6 code walkthrough
outputs/        Week 6 figures, tables, reports
week7/          Model testing, scorer V2, model card, hand-off
week7_test/     Copies of the Week 7 documents
week8/          Final evaluation notebook, package, slides, video script, case study
```

## Reproduce
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# place the 27 raw files in data/ (see data/README.md)
python src/01_data_prep_and_integration.py   # ... through 08
python week7/src/week7_model_testing.py
python week8/src/_nb_runner.py week8/src/week8_final_model_evaluation.py week8/outputs/HC_Week8_Final_Model_Evaluation_Notebook.ipynb
python week8/src/build_week8_docs.py && python week8/src/build_week8_slides.py
```
Read CSVs with `keep_default_na=False, na_values=[""]`: the value "None" is real data.

## Privacy
Names, phone numbers and emergency contacts are excluded from exports. Patient-level data is not stored in this repository.

## Limitations
No attendance data; wrong target (admission); one year of data; look-ahead feature (`PatientVisits`); no external validation. Teammate integration evidence is recorded separately by each intern.

## Reflection
A rigorous negative result is more useful than a model that looks good. Next step: obtain appointment status data and rerun the same tests.
