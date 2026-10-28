# HealthConnect Rwanda — Healthcare Predictive Analytics & Population Health Intelligence

AnalystLab Africa · Experience Lab · **Week 6** · Track 2: Data Science

| Week | Focus | Status |
|---|---|---|
| 6 | Data prep, EDA, statistics, forecasting, models, clustering, reports | ✅ |
| 7 | Model testing, error analysis, refinement, ML-interface hand-off (`week7/`) | ✅ |

## Reproduce
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
# put the 27 raw files in data/ (see data/README.md), then:
python src/01_data_prep_and_integration.py
python src/02_features_and_eda.py
python src/03_statistical_analysis.py
python src/04_forecasting_and_models.py
python src/05_clustering_and_pca.py
python src/06_recommendations_and_export.py
python src/07_build_reports.py
python src/08_build_slides.py
```
Step-by-step guide: `docs/HealthConnect_Week6_Code_Walkthrough.docx`. The same analysis as one notebook: `notebooks/HC_Week6_DataScience_Notebook.ipynb`.

## Headline results
- 20,000 patients, 5,000 appointment encounters, 60,000 bills, 20 facilities; 108 data-quality checks (8 critical).
- Only 5,000 of 60,000 billed visits have an appointment record; facility/department/doctor keys are inconsistent.
- 0.92 bn RWF (9.9%) of billing unpaid; 21,028 claims (2.44 bn RWF) pending; claim approval 50%.
- 0 of 15 hypothesis tests significant; admission model CV AUC ≈ 0.51 (chance); satisfaction R² ≈ 0.
- 2027 forecast: ≈ 4,975 visits/month (MAPE 14.0%), ≈ 1.10 bn RWF revenue.

## Layout
`src/` scripts · `outputs/figures` 36 charts · `outputs/tables` result tables · `outputs/reports` Word reports + slides · `docs/` walkthrough. Raw data is git-ignored; exports contain no names, phone numbers or emergency contacts.

## Week 7
See `week7/README.md`: the Week 6 candidate model fails all 6 pre-registered acceptance criteria (hold-out AUC 0.529, CI 0.47–0.59); seven refinements give no meaningful gain; scoring-interface tests improve from 8/18 to 18/18. Report, summary and testing record are in `week7/outputs/reports/`.
