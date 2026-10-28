"""
HealthConnect Rwanda - Week 7 (Data Science -> ML Engineering hand-off)
Scoring wrapper around the Week 6 candidate admission-risk model.

V1  = the bare model call, exactly as it existed at the end of Week 6.
V2  = the refined version produced after Week 7 testing (input contract, validation, warnings, output schema).

IMPORTANT: the model has no demonstrated predictive skill (see the Week 7 report). Scores must NOT be used for
patient-level decisions. V2 therefore returns an explicit `use_caution` field and the model card records this.
"""
import json
import numpy as np
import pandas as pd

# ---- input contract (what ML Engineering may send) ---------------------------------------------------
CONTRACT = {
    "numeric": {  # name: (min, max, description)
        "Age": (0, 110, "years"), "BMI": (8, 80, "kg/m2"), "RiskScore": (0, 6, "rule-based risk score"),
        "DoctorYearsExp": (0, 60, "years"), "PatientVisits": (0, 50, "appointments on record"),
        "Month_": (1, 12, "calendar month of the appointment"), "ChronicFlag": (0, 1, "0/1"),
    },
    "categorical": {  # name: allowed values (None = free text, only checked for presence)
        "Gender": ["Male", "Female"],
        "MaritalStatus": None, "Education": None, "Occupation": None, "BloodGroup": None,
        "ChronicCondition": ["None", "Asthma", "Hypertension", "Diabetes"],   # "None" is a real value, not a missing value
        "ProvinceName": ["Eastern", "Kigali City", "Northern", "Southern", "Western"],
        "VisitType": None, "DepartmentName": None,
        "FacilityID": None, "DayOfWeek": None,
    },
}
REQUIRED = list(CONTRACT["numeric"]) + list(CONTRACT["categorical"])
MODEL_VERSION = "week7-v2"


class AdmissionScorerV1:
    """Week 6 behaviour: pass the data straight to the fitted pipeline."""
    def __init__(self, pipe):
        self.pipe = pipe

    def predict(self, df):
        return self.pipe.predict_proba(df)[:, 1]


class ScoringError(ValueError):
    """Raised for input that violates the contract; the message always names the field and the rule."""


class AdmissionScorerV2:
    def __init__(self, pipe, train_categories, band_cuts):
        self.pipe, self.train_categories, self.band_cuts = pipe, train_categories, band_cuts

    # -- validation ---------------------------------------------------------------------------------
    def validate(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ScoringError("Input must be a pandas DataFrame with one row per appointment.")
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ScoringError(f"Missing required column(s): {missing}")
        warns = [[] for _ in range(len(df))]
        for c, (lo, hi, _) in CONTRACT["numeric"].items():
            col = pd.to_numeric(df[c], errors="coerce")
            bad_type = col.isna() & df[c].notna()
            if bad_type.any():
                raise ScoringError(f"Column '{c}' must be numeric; non-numeric value(s) at row(s) {list(df.index[bad_type])[:5]}")
            if col.isna().any():
                raise ScoringError(f"Column '{c}' has missing value(s) at row(s) {list(df.index[col.isna()])[:5]}; missing values are not imputed")
            out = (col < lo) | (col > hi)
            if out.any():
                raise ScoringError(f"Column '{c}' out of allowed range [{lo}, {hi}] at row(s) {list(df.index[out])[:5]}")
        for c, allowed in CONTRACT["categorical"].items():
            if df[c].isna().any():
                hint = (" - if the data came from a CSV, the text 'None' may have been parsed as missing; read it with keep_default_na=False"
                        if c == "ChronicCondition" else "")
                raise ScoringError(f"Column '{c}' has missing value(s) at row(s) {list(df.index[df[c].isna()])[:5]}{hint}")
            if allowed is not None:
                bad = ~df[c].isin(allowed)
                if bad.any():
                    raise ScoringError(f"Column '{c}' has value(s) outside the allowed list {allowed}: {sorted(df.loc[bad, c].astype(str).unique())[:5]}")
            else:  # unseen-in-training categories are scored, but flagged
                unseen = ~df[c].astype(str).isin(self.train_categories[c])
                for i in np.where(unseen.values)[0]:
                    warns[i].append(f"'{c}' value '{df[c].iloc[i]}' not seen in training; treated as 'other'")
        extra = [c for c in df.columns if c not in REQUIRED]
        return warns, extra

    # -- scoring ------------------------------------------------------------------------------------
    def predict(self, df):
        if isinstance(df, pd.DataFrame) and len(df) == 0:
            return pd.DataFrame(columns=["risk_score", "risk_band", "warnings", "model_version", "use_caution"])
        warns, extra = self.validate(df)
        p = self.pipe.predict_proba(df[REQUIRED])[:, 1]
        band = np.where(p >= self.band_cuts["high"], "higher", np.where(p <= self.band_cuts["low"], "lower", "typical"))
        w = [("; ".join(x) if x else "") for x in warns]
        if extra:
            w = [(s + "; " if s else "") + f"ignored extra column(s): {extra}" for s in w]
        return pd.DataFrame({"risk_score": p, "risk_band": band, "warnings": w, "model_version": MODEL_VERSION,
                             "use_caution": "No demonstrated predictive skill (AUC ~0.5): do not use for patient-level decisions"}, index=df.index)

    def contract_json(self):
        return json.dumps({"version": MODEL_VERSION, "numeric": CONTRACT["numeric"], "categorical": CONTRACT["categorical"],
                           "output": ["risk_score (0-1)", "risk_band", "warnings", "model_version", "use_caution"]}, indent=2)
