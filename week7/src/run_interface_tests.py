"""
Hand-off test for ML Engineering: run the scoring-interface checks against the saved scorer.
Usage (from the week7 folder):  python src/run_interface_tests.py
Prints PASS/FAIL per check. Copy the output and the date into the cross-track record.
"""
import os, sys
import numpy as np, pandas as pd, joblib
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
import admission_risk_scorer as ars   # needed so joblib can rebuild the scorer class
art = joblib.load("outputs/models/admission_scorer_v2.joblib"); sc = art["scorer"]
print("Model version:", art["model_version"], "| trained with scikit-learn", art["sklearn_version"], "| rows", art["trained_on_rows"])
E = pd.read_csv("../outputs/data/HC_Integrated_Encounters_5000.csv", keep_default_na=False, na_values=[""])
E["Month_"] = E.MonthNum; X = E[ars.REQUIRED].copy(); X["FacilityID"] = X.FacilityID.astype(str)
row = X.iloc[[0]].copy(); res = []
def check(name, ok, detail=""): res.append(ok); print(("PASS " if ok else "FAIL ") + name + (f"  - {detail}" if detail else ""))
def raises(df, field):
    try: sc.predict(df); return False, "no error"
    except Exception as e: return field in str(e), str(e)[:90]
o = sc.predict(X.iloc[:200]); check("valid batch scored", len(o) == 200 and o.risk_score.between(0, 1).all())
check("output schema", all(c in o.columns for c in ["risk_score", "risk_band", "warnings", "model_version", "use_caution"]))
check("column order independent", np.allclose(sc.predict(X.iloc[:200][X.columns[::-1]]).risk_score, o.risk_score))
d = row.copy(); d["Age"] = -5; check("rejects Age = -5", *raises(d, "Age"))
d = row.copy(); d["ProvinceName"] = "Mars"; check("rejects ProvinceName = Mars", *raises(d, "ProvinceName"))
d = row.copy(); d["Age"] = np.nan; check("rejects missing Age", *raises(d, "Age"))
d = row.copy(); d["ChronicCondition"] = np.nan; check("rejects missing ChronicCondition (CSV 'None' pitfall)", *raises(d, "ChronicCondition"))
check("missing column named in error", raises(row.drop(columns=["Age"]), "Age")[0])
d = row.copy(); d["PatientName"] = "x"; check("extra column reported", "PatientName" in " ".join(sc.predict(d).warnings))
check("empty input returns empty result", len(sc.predict(X.iloc[0:0])) == 0)
print(f"\n{sum(res)} of {len(res)} checks passed")
sys.exit(0 if all(res) else 1)
