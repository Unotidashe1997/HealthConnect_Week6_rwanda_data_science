# Week 7 — Model Testing, Error Analysis & Refinement (Data Science track)

Continues the Week 6 pipeline: tests the Week 6 candidate admission classifier, refines it on evidence, re-tests it, and tests its scoring interface for ML Engineering.

| File | Purpose |
|---|---|
| `src/week7_model_testing.py` | Analysis (percent-format notebook source): criteria, stability, thresholds, calibration, error analysis, segments, overfitting, feature validity, power test, refinement, interface tests |
| `src/admission_risk_scorer.py` | Scoring wrapper: V1 (Week 6 behaviour) and V2 (validated, with input/output contract) |
| `src/build_week7_docs.py` | Builds the report, project summary and testing record from the results |
| `src/run_interface_tests.py` | Hand-off checks for ML Engineering |
| `outputs/HC_Week7_Model_Testing_Notebook.ipynb` | Executed notebook |
| `outputs/reports/` | Model Testing & Refinement Report, Project Summary, Testing & Validation Record (xlsx) |
| `outputs/figures/`, `outputs/tables/` | 8 charts, result tables |
| `outputs/model_card.md`, `outputs/input_output_contract.json`, `outputs/models/` | Hand-off package |

**Run:** Week 6 scripts 01–06 first (creates `outputs/data/HC_Integrated_Encounters_5000.csv`), then from `week7/`:
```bash
python src/_nb_runner.py src/week7_model_testing.py outputs/HC_Week7_Model_Testing_Notebook.ipynb
python src/build_week7_docs.py
python src/run_interface_tests.py
```
**Headline:** 0 of 6 pre-registered acceptance criteria met (hold-out AUC 0.529, CI 0.47–0.59); 7 refinements gave no meaningful improvement; scoring-interface tests 8/18 → 18/18 after refinement. The project data contain no attendance field, so a missed-appointment model needs the attendance dataset.
