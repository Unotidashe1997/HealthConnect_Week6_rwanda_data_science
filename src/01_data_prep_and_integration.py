"""
HealthConnect Rwanda - Week 6 (Data Science track)
Part 1 - Load, clean, validate and integrate the data; Data Quality checks

Run from the repository root:  python src/01_data_prep_and_integration.py
"""
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
import matplotlib
matplotlib.use("Agg")
import warnings, os, re, json, textwrap, math, itertools
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib as mpl
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from itertools import product
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold, cross_validate, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, r2_score, mean_absolute_error, roc_curve
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from _state import restore, save
restore(globals(), __file__)

# %% [markdown]
# # HealthConnect Rwanda — Healthcare Predictive Analytics & Population Health Intelligence
# **AnalystLab Africa · Experience Lab · Week 6 · Track 2 — Data Science**
#
# This notebook covers every section of the Data Science brief, in order:
# 1. Data preparation (loading, cleaning, validation, integration) · 2. Feature engineering ·
# 3. Exploratory analysis · 4. Statistical analysis · 5. Predictive analytics & forecasting ·
# 6. Clustering & PCA · 7. Business recommendations.
#
# **Reading guide.** The 27 source datasets are *relationally inconsistent* in several important ways
# (documented in section 1.4). Every analysis states which records it uses and how many, and every
# interpretation caption is generated from the computed numbers, so conclusions always match the evidence.

# %%
import warnings, os, re, json, textwrap, math, itertools
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib as mpl
import seaborn as sns
from scipy import stats
warnings.filterwarnings("ignore")

DATA, OUT = "data/", "outputs/"
FIGDIR = OUT + "figures/"
os.makedirs(FIGDIR, exist_ok=True)
SEED = 42
rng = np.random.default_rng(SEED)
AS_OF = pd.Timestamp("2026-10-23")         # analysis date

# --- results registries used later by the reports -------------------------------------------
R = {}        # headline numbers
FIGS = {}     # figure name -> {title, interp}
DQ = []       # data-quality findings

# --- visual identity (validated categorical palette, fixed order; muted ink; hairline grid) ---
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SEQ = mpl.colors.LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
DIV = mpl.colors.LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#eb6834"])
mpl.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11.5, "axes.titleweight": "semibold",
    "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.frameon": False,
    "legend.fontsize": 8.5, "font.size": 9.5, "figure.dpi": 100, "axes.axisbelow": True,
    "axes.prop_cycle": mpl.cycler(color=C),
})


def save_fig(fig, name, interp, title=None):
    """Attach a plain-language interpretation under the chart, save at 150 dpi, register for the reports."""
    title = title or (fig._suptitle.get_text() if fig._suptitle else fig.axes[0].get_title())
    wrapped = "\n".join(textwrap.wrap("Interpretation: " + interp, 150))
    fig.text(0.01, -0.015, wrapped, ha="left", va="top", fontsize=8.3, color=INK2, linespacing=1.35)
    fig.savefig(f"{FIGDIR}{name}.png", dpi=150, bbox_inches="tight", facecolor=SURF)
    FIGS[name] = {"title": title, "interp": interp}


def dq(table, check, n_checked, n_issue, severity, action, note=""):
    DQ.append(dict(Table=table, Check=check, Records_Checked=int(n_checked), Issues_Found=int(n_issue),
                   Pct_Affected=round(100 * n_issue / n_checked, 2) if n_checked else 0.0,
                   Severity=severity, Action_Taken=action, Note=note))


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


print("pandas", pd.__version__, "| numpy", np.__version__)

# %% [markdown]
# ## 1. Data preparation
# ### 1.1 Task 1 — Load every enterprise dataset
# Lookup/master tables and flat files are read from the CSV extracts; the two Excel workbooks hold the
# clinical transactions (6 sheets) and a copy of the facility master data (6 sheets). `keep_default_na=False`
# is essential: `ChronicCondition` contains the literal text **"None"** (= no chronic disease), which pandas
# would otherwise silently convert to *missing*.

# %%
def rd(f, **kw):
    return pd.read_csv(DATA + f, encoding="utf-8-sig", keep_default_na=False, na_values=[""], **kw)

D = {
    "Countries": rd("HCCountries.csv"), "Provinces": rd("HC_Provinces.csv"), "Districts": rd("HC_Districts.csv"),
    "Sectors": rd("HC_Sectors.csv"), "Villages": rd("HC_Villages.csv"), "Ownership_Types": rd("HC_Ownership_Types.csv"),
    "Departments": rd("HC_Departments.csv"), "Medical_Specialties": rd("HC_Medical_Specialties.csv"),
    "Services": rd("HC_Services.csv"), "Medical_Equipment": rd("HC_Medical_Equipment.csv"),
    "Rooms": rd("HC_Rooms.csv"), "Beds": rd("HC_Beds.csv"),
    "Doctors": rd("HC_Doctor.csv"), "Nurses": rd("HC_Nurse.csv"), "Pharmacists": rd("Pharmacist.csv"),
    "Lab_Scientists": rd("Laboratory_Scientist.csv"), "Radiographers": rd("HC_Radiographer.csv"),
    "Admin_Staff": rd("HC_Administrative_Staff.csv"),
    "Patients": rd("HC_Patients.csv"), "Diagnoses_csv": rd("HC_Diagnoses.csv"),
    "Billing": rd("HC_Billing.csv"), "Insurance_Claims": rd("Insurance_Claims.csv"),
    "Payments": rd("HC_Payments.csv"), "Patient_Feedback": rd("HC_Patient_Feedback.csv"),
    "Revenue_Summary": rd("HC_Revenue_Summary.csv"),
}
xl = pd.ExcelFile(DATA + "HC_Clinical_Transactions.xlsx")
T = {s: xl.parse(s) for s in xl.sheet_names}                      # Appointments, Diagnoses, Laboratory_Tests, ...
fm = pd.ExcelFile(DATA + "HC_Facility_Master_Data.xlsx")
FM = {s: fm.parse(s) for s in fm.sheet_names}

# the Facility-Master workbook should simply duplicate the CSV masters -> verify
fm_map = {"Departments": "Departments", "Medical_Specialties": "Medical_Specialties", "Services": "Services",
          "Medical_Equipment": "Medical_Equipment", "Rooms": "Rooms", "Beds": "Beds"}
fm_same = {k: FM[k].astype(str).reset_index(drop=True).equals(D[v].astype(str).reset_index(drop=True)) for k, v in fm_map.items()}
print("Facility-master workbook identical to CSV masters:", fm_same)

inv = [(k, v.shape[0], v.shape[1], "csv") for k, v in D.items()] + [(f"Clinical_Transactions::{k}", v.shape[0], v.shape[1], "xlsx") for k, v in T.items()]
inventory = pd.DataFrame(inv, columns=["Dataset", "Rows", "Columns", "Source"])
R["n_source_files"] = 20 + 2  # 20 csv (excl. pdf) + 2 xlsx
R["n_tables_loaded"] = len(inventory)
print(f"{len(inventory)} tables loaded; total rows = {inventory.Rows.sum():,}")
inventory

# %% [markdown]
# ### 1.2 Task 1 — Relationships between datasets
# The data model is a hierarchy **Country → Province → District → Sector → Village → Patient**, a facility
# structure **Facility → Department → (Rooms → Beds, Equipment, Services, Specialties, Staff)** and a
# transaction chain **Appointment → (Diagnosis, Lab test, Prescription, Admission, Referral, Billing → Payment / Claim, Feedback)**.
# The next cell tests every join key and reports match rates.

# %%
ap = T["Appointments"].copy()
pat, dept, doc = D["Patients"], D["Departments"], D["Doctors"]
fac_ids = set(dept.FacilityID)

def fk(child, ckey, parent_keys, label):
    ok = child[ckey].isin(parent_keys)
    return dict(Relationship=label, Child_Rows=len(child), Matched=int(ok.sum()), Match_Rate=round(ok.mean(), 4))

rel = [
    fk(D["Provinces"], "CountryID", D["Countries"].CountryID, "Province → Country"),
    fk(D["Districts"], "ProvinceID", D["Provinces"].ProvinceID, "District → Province"),
    fk(D["Sectors"], "DistrictID", D["Districts"].DistrictID, "Sector → District"),
    fk(D["Villages"], "SectorID", D["Sectors"].SectorID, "Village → Sector"),
    fk(pat, "VillageID", D["Villages"].VillageID, "Patient → Village"),
    fk(dept, "FacilityID", fac_ids, "Department → Facility (ID list derived from Departments)"),
    fk(D["Rooms"], "DepartmentID", dept.DepartmentID, "Room → Department"),
    fk(D["Beds"], "RoomID", D["Rooms"].RoomID, "Bed → Room"),
    fk(D["Medical_Equipment"], "DepartmentID", dept.DepartmentID, "Equipment → Department"),
    fk(D["Services"], "DepartmentID", dept.DepartmentID, "Service → Department"),
    fk(D["Medical_Specialties"], "DepartmentID", dept.DepartmentID, "Specialty → Department"),
]
for nm in ["Doctors", "Nurses", "Pharmacists", "Lab_Scientists", "Radiographers", "Admin_Staff"]:
    rel.append(fk(D[nm], "DepartmentID", dept.DepartmentID, f"{nm} → Department"))
rel += [
    fk(ap, "PatientID", pat.PatientID, "Appointment → Patient"),
    fk(ap, "DoctorID", doc.StaffID, "Appointment → Doctor"),
    fk(ap, "DepartmentID", dept.DepartmentID, "Appointment → Department"),
    fk(ap, "FacilityID", fac_ids, "Appointment → Facility"),
]
for s in ["Diagnoses", "Laboratory_Tests", "Prescriptions", "Admissions", "Referrals"]:
    rel.append(fk(T[s], "AppointmentID", ap.AppointmentID, f"{s} (xlsx) → Appointment"))
rel += [
    fk(T["Referrals"], "ReferredFacilityID", fac_ids, "Referral → Referred facility"),
    fk(D["Diagnoses_csv"], "AppointmentID", ap.AppointmentID, "Diagnoses (csv) → Appointment"),
    fk(D["Billing"], "AppointmentID", ap.AppointmentID, "Billing → Appointment"),
    fk(D["Payments"], "BillingID", D["Billing"].BillingID, "Payment → Billing"),
    fk(D["Insurance_Claims"], "BillingID", D["Billing"].BillingID, "Claim → Billing"),
    fk(D["Patient_Feedback"], "AppointmentID", ap.AppointmentID, "Feedback → Appointment"),
]
REL = pd.DataFrame(rel)
R["relationships"] = REL.to_dict("records")
REL

# %% [markdown]
# **Key structural finding.** The Appointments sheet holds only **5,000** appointments (APT000001–APT005000),
# while Billing, the Diagnoses CSV and Patient Feedback each describe ~**60,000** appointments (APT000001–APT060000).
# Only the first 5,000 can be linked to patient, doctor, facility and department. The analysis therefore uses two
# integrated datasets: a **clinical encounter master** (5,000 appointments, full detail) and a **financial/feedback
# universe** (60,000 bills, with diagnosis & feedback but no patient/facility). Section 1.6 builds both.

# %% [markdown]
# ### 1.3 Task 2 — Data cleaning
# Steps: (a) missing-value detection, (b) duplicate removal, (c) invalid-record detection, (d) data-type conversion,
# (e) date formatting, (f) category standardisation. Records that are *implausible but not provably wrong* are
# **flagged** rather than deleted, so no information is lost and sensitivity can be tested.

# %%
# (a) Missing values --------------------------------------------------------------------------------------
miss = []
for k, df in list(D.items()) + [(f"CT::{k}", v) for k, v in T.items()]:
    n = df.isna().sum()
    for c, v in n[n > 0].items():
        miss.append((k, c, int(v), round(100 * v / len(df), 2)))
MISS = pd.DataFrame(miss, columns=["Table", "Column", "Missing", "Pct"])
print("True missing values after reading 'None' as a real category:")
print(MISS if len(MISS) else "  none in any of the 31 tables")
chronic_none = (pat.ChronicCondition == "None").sum()
dq("Patients", "ChronicCondition 'None' text mis-read as missing by default CSV parsers", len(pat), chronic_none, "Medium",
   "Read with keep_default_na=False; kept as explicit category 'None' (= no chronic disease)",
   "Naïve read shows 25.4% 'missing' values that are really a valid answer")
dq("All tables", "Missing values (true NaN) across 31 tables", int(inventory.Rows.sum()), int(MISS.Missing.sum()), "Info",
   "None required", "Tables are structurally complete — the problems are relational, not nulls")

# (b) Duplicates ------------------------------------------------------------------------------------------
keys = {"Patients": ("Patients", "PatientID"), "Appointments": (ap, "AppointmentID"), "Billing": ("Billing", "BillingID"),
        "Payments": ("Payments", "PaymentID"), "Insurance_Claims": ("Insurance_Claims", "ClaimID"),
        "Patient_Feedback": ("Patient_Feedback", "FeedbackID"), "Diagnoses_csv": ("Diagnoses_csv", "DiagnosisID")}
for nm, (src, key) in keys.items():
    df = D[src] if isinstance(src, str) else src
    dq(nm, f"Duplicate primary key ({key})", len(df), df[key].duplicated().sum(), "Info", "None required" if not df[key].duplicated().any() else "Dropped")
for s in ["Diagnoses", "Laboratory_Tests", "Prescriptions", "Admissions", "Referrals"]:
    k = T[s].columns[0]
    dq(f"CT::{s}", f"Duplicate primary key ({k})", len(T[s]), T[s][k].duplicated().sum(), "Info", "None required")
    dq(f"CT::{s}", "More than one record per appointment", len(T[s]), T[s].AppointmentID.duplicated().sum(), "Info", "None required")
for nm in ["Doctors", "Nurses", "Pharmacists", "Lab_Scientists", "Radiographers", "Admin_Staff"]:
    dq(nm, "Duplicate StaffID", len(D[nm]), D[nm].StaffID.duplicated().sum(), "Info", "None required")
dq("Billing", "Appointments billed more than once", len(D["Billing"]), D["Billing"].AppointmentID.duplicated().sum(), "Info", "None required")
full_dup = sum(df.duplicated().sum() for df in D.values()) + sum(df.duplicated().sum() for df in T.values())
dq("All tables", "Fully duplicated rows", int(inventory.Rows.sum()), int(full_dup), "Info", "None required")
dup_visit = ap.duplicated(["PatientID", "DoctorID", "Date"]).sum()
dq("Appointments", "Same patient + doctor + date appears twice", len(ap), dup_visit, "Info", "None required")
print("Duplicate-key checks logged. Duplicates removed: 0 (none exist). Full-row duplicates:", full_dup)

# %%
# (c)-(f) Patients: types, dates, categories, invalid/implausible records ---------------------------------
P = pat.copy()
P["DateOfBirth"] = pd.to_datetime(P["DateOfBirth"], format="%m/%d/%Y", errors="coerce")
dq("Patients", "Unparseable DateOfBirth", len(P), P.DateOfBirth.isna().sum(), "Info", "Converted to datetime (format m/d/Y)")
dob_jan1 = ((P.DateOfBirth.dt.month == 1) & (P.DateOfBirth.dt.day == 1)).sum()
dq("Patients", "DateOfBirth is always 1 January (placeholder day/month)", len(P), dob_jan1, "Medium",
   "Age used as supplied (equals 2026 − birth year); month-level age precision is unavailable",
   "Age is accurate only to ±1 year; infants cannot be separated from 1-year-olds")
age_chk = ((2026 - P.DateOfBirth.dt.year) != P.Age).sum()
dq("Patients", "Age inconsistent with DateOfBirth (reference year 2026)", len(P), age_chk, "Info", "None required")
dq("Patients", "Impossible age (<0 or >110)", len(P), ((P.Age < 0) | (P.Age > 110)).sum(), "Info", "None required",
   f"Age range {P.Age.min()}–{P.Age.max()} (mean {P.Age.mean():.1f})")
P["BMI_calc"] = (P.WeightKg / (P.HeightCm / 100) ** 2).round(1)
dq("Patients", "Stored BMI differs from Weight/Height² by >0.2", len(P), (abs(P.BMI - P.BMI_calc) > 0.2).sum(), "Info", "None required")

# phone numbers were exported in scientific notation (2.50788E+11) -> only 10 distinct values survive
P["Phone"] = P["Phone"].astype(str)
n_phone_unique = P.Phone.nunique()
dq("Patients", "Phone numbers truncated by scientific-notation export (2.50788E+11)", len(P), len(P) - n_phone_unique, "High",
   "Phone excluded from analysis & integrated dataset; flagged PhoneValid = False",
   f"Only {n_phone_unique} distinct values remain for 20,000 patients, so 'duplicate phone' detection is meaningless; "
   "the source export must be re-run with phone stored as text")
P["PhoneValid"] = False

# category standardisation: trim + consistent case; then verify
for c in P.select_dtypes(include=["object", "string"]).columns:
    if c in ("PatientID", "VillageID", "InsuranceID", "Phone"):
        P[c] = P[c].astype(str).str.strip()
    else:
        P[c] = P[c].astype(str).str.strip()
cat_cols = ["Gender", "MaritalStatus", "Occupation", "Education", "BloodGroup", "ChronicCondition", "Status"]
std_issue = {c: int((D["Patients"][c].astype(str) != D["Patients"][c].astype(str).str.strip().str.title()).sum()) for c in ["Gender", "MaritalStatus", "Occupation", "Education", "ChronicCondition", "Status"]}
dq("Patients", "Category labels needing trim/case standardisation", len(P) * len(std_issue), sum(std_issue.values()), "Info",
   "Trim/strip applied to all text columns; labels were already consistent")
print("Distinct categories —", {c: P[c].nunique() for c in cat_cols})

P["ChronicFlag"] = (P.ChronicCondition != "None").astype(int)

# invalid / implausible record detection (flag, don't delete)
P["Flag_AnthroImplausible"] = (((P.Age <= 5) & ((P.HeightCm >= 125) | (P.WeightKg >= 35))) | ((P.Age <= 10) & (P.HeightCm >= 150)))
P["Flag_BMIExtreme"] = (P.BMI < 13) | (P.BMI > 50)
P["Flag_AgeLogic"] = ((P.Age < 16) & (P.MaritalStatus != "Single")) | ((P.Age < 18) & P.Education.isin(["Diploma", "Bachelor", "Master"])) | ((P.Age < 15) & (P.Occupation != "Student"))
for f, desc, sev in [("Flag_AnthroImplausible", "Child with adult height/weight (age ≤5 & ≥125 cm or ≥35 kg; age ≤10 & ≥150 cm)", "High"),
                     ("Flag_BMIExtreme", "BMI < 13 or > 50 (physiologically extreme)", "Medium"),
                     ("Flag_AgeLogic", "Age-illogical demographics (child married/widowed, minor with Diploma/Bachelor/Master, under-15 with a job)", "High")]:
    dq("Patients", desc, len(P), int(P[f].sum()), sev, f"Flagged ({f}); retained and sensitivity-tested",
       "Values look randomly generated independently of age — treat demographic cross-tabs with caution")
P["Flag_AnyPatientIssue"] = P[["Flag_AnthroImplausible", "Flag_BMIExtreme", "Flag_AgeLogic"]].any(axis=1)
dq("Patients", "Status = Active for every patient (zero variance; no inactive/deceased records)", len(P), len(P), "Low", "Noted; 'Active Patients' = all patients")
for c in ["Flag_AnthroImplausible", "Flag_BMIExtreme", "Flag_AgeLogic", "Flag_AnyPatientIssue"]:
    print(f"{c:26s}: {P[c].sum():6,d}  ({pct(P[c].mean())})")

# %%
# Appointments: dates, future dates, categories ---------------------------------------------------------
ap["Date"] = pd.to_datetime(ap["Date"], errors="coerce")
dq("Appointments", "Unparseable Date", len(ap), ap.Date.isna().sum(), "Info", "Converted to datetime (ISO)")
ap["Month"] = ap.Date.dt.to_period("M").dt.to_timestamp()
future = (ap.Date > AS_OF).sum()
dq("Appointments", f"Appointments dated after the analysis date ({AS_OF.date()})", len(ap), future, "Medium",
   "Retained (could be scheduled bookings); full calendar-year 2026 treated as the observed history",
   "No appointment-status column exists, so booked / attended / cancelled cannot be distinguished")
dq("Appointments", "Cancellation / no-show status column missing", len(ap), len(ap), "Medium", "Cancellation-rate KPI cannot be computed", "Recommend adding AppointmentStatus")
print("Appointment dates:", ap.Date.min().date(), "→", ap.Date.max().date(), "| future-dated:", future)

# category standardisation for transactional tables
for nm, df, cols in [("Appointments", ap, ["VisitType"]), ("Diagnoses_csv", D["Diagnoses_csv"], ["Diagnosis", "Severity"]),
                     ("Payments", D["Payments"], ["Method"]), ("Billing", D["Billing"], ["Status"]),
                     ("Insurance_Claims", D["Insurance_Claims"], ["Provider", "Status"]), ("Patient_Feedback", D["Patient_Feedback"], ["Comment"])]:
    bad = 0
    for c in cols:
        s = df[c].astype(str)
        bad += int((s != s.str.strip()).sum())
        df[c] = s.str.strip()
    dq(nm, f"Category labels with stray whitespace ({', '.join(cols)})", len(df) * len(cols), bad, "Info", "Trimmed; none found")
print("Category label sets:", {k: sorted(D[k][c].unique()) for k, c in [("Payments", "Method"), ("Insurance_Claims", "Provider"), ("Insurance_Claims", "Status"), ("Billing", "Status")]})

# %% [markdown]
# ### 1.4 Task 3 — Data validation
# Validation covers the seven required checks (foreign keys, duplicate Patient IDs, duplicate Appointment IDs, invalid
# Facility IDs, invalid Department IDs, impossible ages, invalid billing amounts) plus cross-field consistency checks
# that the structure of this database makes necessary.

# %%
bill, pay, clm, fb = D["Billing"], D["Payments"], D["Insurance_Claims"], D["Patient_Feedback"]
# --- the seven required checks --------------------------------------------------------------------------
for _, r in REL.iterrows():
    miss_n = r.Child_Rows - r.Matched
    if r.Match_Rate == 1:
        sev = "Info"; act = "None required"
    else:
        sev = "Critical" if r.Match_Rate < 0.5 else "High"
        act = "Orphans retained, excluded from joined analyses; cohort-based analyses disclose n"
    dq(r.Relationship.split(" → ")[0], f"Foreign key: {r.Relationship}", r.Child_Rows, miss_n, sev, act,
       "Child records whose parent appointment is absent from the 5,000-row Appointments sheet" if miss_n and "Appointment" in r.Relationship else "")
dq("Patients", "Duplicate Patient IDs", len(pat), pat.PatientID.duplicated().sum(), "Info", "None required")
dq("Appointments", "Duplicate Appointment IDs", len(ap), ap.AppointmentID.duplicated().sum(), "Info", "None required")
dq("Appointments", "Invalid Facility IDs (not in master)", len(ap), (~ap.FacilityID.isin(fac_ids)).sum(), "Info", "None required")
dq("Appointments", "Invalid Department IDs (not in master)", len(ap), (~ap.DepartmentID.isin(dept.DepartmentID)).sum(), "Info", "None required")
dq("Billing", "Invalid billing amounts (≤ 0 or missing)", len(bill), ((bill.Amount <= 0) | bill.Amount.isna()).sum(), "Info", "None required",
   f"Amounts range {bill.Amount.min():,.0f}–{bill.Amount.max():,.0f} RWF")
dq("Payments", "Invalid payment amounts (≤ 0)", len(pay), (pay.AmountPaid <= 0).sum(), "Info", "None required")

# --- cross-field consistency -------------------------------------------------------------------------------
dmap_fac = dept.set_index("DepartmentID").FacilityID
dmap_name = dept.set_index("DepartmentID").DepartmentName
ap["Dept_Facility"] = ap.DepartmentID.map(dmap_fac)
ap["DepartmentName"] = ap.DepartmentID.map(dmap_name)
mism = (ap.Dept_Facility != ap.FacilityID).sum()
dq("Appointments", "Appointment FacilityID ≠ facility that owns the appointment's DepartmentID", len(ap), mism, "Critical",
   "Facility = appointment.FacilityID; department TYPE = name of DepartmentID (documented assumption)",
   "95% of appointments reference a department belonging to a different facility — IDs were generated independently")
dd = doc.set_index("StaffID")
ap["Doc_Facility"], ap["Doc_DeptID"], ap["Doc_DeptName"] = ap.DoctorID.map(dd.FacilityID), ap.DoctorID.map(dd.DepartmentID), ap.DoctorID.map(dd.Department)
dq("Appointments", "Doctor's FacilityID ≠ appointment FacilityID", len(ap), (ap.Doc_Facility != ap.FacilityID).sum(), "Critical",
   "Doctor workload computed per DoctorID; facility attribution uses appointment.FacilityID")
dq("Appointments", "Doctor's department name ≠ appointment department name", len(ap), (ap.Doc_DeptName != ap.DepartmentName).sum(), "High",
   "Noted; department of the appointment record used")
staff_all = []
for nm in ["Doctors", "Nurses", "Pharmacists", "Lab_Scientists", "Radiographers", "Admin_Staff"]:
    s = D[nm].copy(); s["Cadre"] = nm; staff_all.append(s)
STAFF = pd.concat(staff_all, ignore_index=True)
STAFF["Dept_Facility"] = STAFF.DepartmentID.map(dmap_fac)
STAFF["DeptNameFromID"] = STAFF.DepartmentID.map(dmap_name)
dq("Staff (all cadres)", "Staff FacilityID ≠ facility owning their DepartmentID", len(STAFF), (STAFF.Dept_Facility != STAFF.FacilityID).sum(), "High",
   "Staff attributed to staff.FacilityID for facility counts and to DepartmentID name for department counts")
dq("Staff (all cadres)", "Staff department name ≠ name of their DepartmentID", len(STAFF), (STAFF.Department != STAFF.DeptNameFromID).sum(), "Info", "None required")
spec = STAFF.Cadre.isin(["Pharmacists", "Lab_Scientists", "Radiographers"])
mis_dep = ((STAFF.Cadre == "Pharmacists") & (STAFF.Department != "Pharmacy")) | ((STAFF.Cadre == "Lab_Scientists") & (STAFF.Department != "Laboratory")) | ((STAFF.Cadre == "Radiographers") & (STAFF.Department != "Radiology"))
dq("Staff (all cadres)", "Pharmacists / lab scientists / radiographers assigned to a department that does not match their cadre (e.g. radiographer in Pharmacy)", int(spec.sum()), int(mis_dep.sum()),
   "Medium", "Staff counted by cadre, not by department fit", "Department assignment appears cyclic/arbitrary")
sp = D["Medical_Specialties"].set_index("DepartmentID").Specialty
dq("Doctors", "Doctor specialty ≠ specialty defined for their DepartmentID", len(doc), (doc.Specialty != doc.DepartmentID.map(sp)).sum(), "Info", "None required")
ref = T["Referrals"].merge(ap[["AppointmentID", "FacilityID"]], on="AppointmentID")
dq("Referrals", "Referred to the same facility the patient was seen at (self-referral)", len(ref), (ref.ReferredFacilityID == ref.FacilityID).sum(), "Low",
   "Retained", f"Only {len(ref):,} of {len(T['Referrals']):,} referrals link to the Appointments sheet")
dq("Beds", "Every bed has Status = 'Available' (no occupied/maintenance state)", len(D["Beds"]), len(D["Beds"]), "High",
   "Bed occupancy estimated indirectly from admissions length-of-stay", "Snapshot status carries no occupancy information")
dq("Equipment", "No quantity, condition or availability fields (1 generic item per department)", len(D["Medical_Equipment"]), len(D["Medical_Equipment"]), "Medium",
   "Equipment distribution reported; availability cannot be measured")
dq("Facilities", "No Health-Facility master table (name, type, ownership, district) is included; OwnershipID is not referenced anywhere", len(fac_ids), len(fac_ids), "High",
   "Facilities identified only as HF001–HF020; ownership/type/regional analysis not possible", "Brief lists a Health Facilities dataset; the file supplied contains only departments, rooms, beds, etc.")
dq("Patients", "No patient registration date and no patient→facility link", len(pat), len(pat), "Medium",
   "Patient growth proxied by first-appointment month (appointment cohort only)")
dq("Patients", "InsuranceID has 15 codes but no insurance-plan lookup table; 100% of patients carry an InsuranceID", len(pat), len(pat), "High",
   "'Insured vs uninsured' comparison replaced by 'claim filed vs no claim' and by claim coverage ratio")

# %%
# Financial consistency ----------------------------------------------------------------------------------
F = bill.merge(pay, on="BillingID", how="left").merge(clm.rename(columns={"Status": "ClaimStatus"}), on="BillingID", how="left")
F["Outstanding"] = F.Amount - F.AmountPaid
F["PaidRatio"] = F.AmountPaid / F.Amount
F["HasClaim"] = F.ClaimID.notna().astype(int)
F["ClaimRatio"] = F.ClaimAmount / F.Amount
dq("Payments", "AmountPaid exceeds billed Amount", len(F), (F.AmountPaid > F.Amount).sum(), "Info", "None required")
dq("Insurance_Claims", "ClaimAmount exceeds billed Amount", len(clm), (F.ClaimAmount > F.Amount).sum(), "Info", "None required")
st_paid_short = ((F.Status == "Paid") & (F.PaidRatio < 1)).sum()
st_pend_full = ((F.Status == "Pending") & (F.PaidRatio == 1)).sum()
st_part_full = ((F.Status == "Partially Paid") & (F.PaidRatio == 1)).sum()
dq("Billing/Payments", "Bill Status 'Paid' but payment < billed amount", (F.Status == "Paid").sum(), st_paid_short, "High",
   "Outstanding balance computed from amounts (Amount − AmountPaid), not from Status")
dq("Billing/Payments", "Bill Status 'Pending' but payment = full amount", (F.Status == "Pending").sum(), st_pend_full, "High", "As above")
dq("Billing/Payments", "Bill Status 'Partially Paid' but payment = full amount", (F.Status == "Partially Paid").sum(), st_part_full, "High", "As above",
   "Status is statistically unrelated to the amount actually paid → treat Status as unreliable")
ins_nc = ((F.Method == "Insurance") & (F.HasClaim == 0)).sum()
dq("Payments/Claims", "Payment method 'Insurance' with no insurance claim on file", (F.Method == "Insurance").sum(), ins_nc, "Medium", "Noted")
rs_tot, bill_tot = D["Revenue_Summary"].Revenue.sum(), bill.Amount.sum()
dq("Revenue_Summary", "Revenue summary total vs sum of billed amounts (should reconcile)", 1, 1, "High",
   "Two revenue series reported separately; forecasting uses the monthly Revenue_Summary (only dated series)",
   f"Summary revenue = {rs_tot:,.0f} RWF vs billed {bill_tot:,.0f} RWF vs paid {pay.AmountPaid.sum():,.0f} RWF (ratio {bill_tot / rs_tot:.1f}x)")
dq("Revenue_Summary", "Summary appointment count vs billed appointments", 1, 1, "Medium", "Noted",
   f"Summary = {D['Revenue_Summary'].Appointments.sum():,} appointments vs {len(bill):,} bills vs {len(ap):,} rows in Appointments sheet")
# feedback rating/comment contradictions
fb["Contradiction"] = ((fb.Rating <= 2) & fb.Comment.isin(["Good", "Excellent"])) | ((fb.Rating >= 4) & (fb.Comment == "Poor"))
dq("Patient_Feedback", "Rating contradicts comment (rating ≤2 with 'Good/Excellent', or ≥4 with 'Poor')", len(fb), fb.Contradiction.sum(), "High",
   "Both fields retained; rating used as the primary satisfaction measure", "Rating and Comment are statistically independent (see χ² test in section 4)")
print(f"Billing status vs payment: Paid-but-short {st_paid_short:,} | Pending-but-fully-paid {st_pend_full:,} | Partial-but-fully-paid {st_part_full:,}")
print(f"Revenue_Summary total {rs_tot:,.0f} | billed {bill_tot:,.0f} | paid {pay.AmountPaid.sum():,.0f}")

# %% [markdown]
# ### 1.5 Data-quality summary

# %%
DQT = pd.DataFrame(DQ)
order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Info": 4}
DQT["_o"] = DQT.Severity.map(order)
DQT = DQT.sort_values(["_o", "Pct_Affected"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)
R["dq_counts"] = DQT.Severity.value_counts().to_dict()
R["dq_n_checks"] = len(DQT)
print("Checks run:", len(DQT), "|", R["dq_counts"])
DQT[DQT.Severity.isin(["Critical", "High", "Medium"])][["Table", "Check", "Records_Checked", "Issues_Found", "Pct_Affected", "Severity", "Action_Taken"]].head(40)

# %% [markdown]
# ### 1.6 Task 1 (cont.) — Build the cleaned, integrated datasets
# * **`encounters`** (5,000 rows) – one row per appointment with patient, geography, doctor, department, diagnosis,
#   lab, prescription, admission, referral, billing, payment, claim and feedback attached (left joins on `AppointmentID`).
# * **`fin`** (60,000 rows) – one row per bill with payment, claim, diagnosis and feedback; facility/department are attached
#   where the appointment exists.
# * Direct identifiers (names, phone, emergency contact) are **removed** from the integrated data (privacy by design).

# %%
geo = (D["Villages"].merge(D["Sectors"], on="SectorID").merge(D["Districts"], on="DistrictID")
       .merge(D["Provinces"], on="ProvinceID")[["VillageID", "SectorID", "DistrictID", "DistrictName", "ProvinceID", "ProvinceName"]])
Pg = P.merge(geo, on="VillageID", how="left")
pat_cols = ["PatientID", "Gender", "Age", "DateOfBirth", "MaritalStatus", "Occupation", "Education", "InsuranceID", "BloodGroup", "HeightCm",
            "WeightKg", "BMI", "ChronicCondition", "ChronicFlag", "VillageID", "SectorID", "DistrictID", "DistrictName", "ProvinceID", "ProvinceName",
            "Flag_AnthroImplausible", "Flag_BMIExtreme", "Flag_AgeLogic", "Flag_AnyPatientIssue"]
PATIENTS = Pg[pat_cols]

dx = T["Diagnoses"].rename(columns={"Diagnosis": "Diagnosis", "Severity": "Severity"})
lab = T["Laboratory_Tests"].rename(columns={"Test": "LabTest", "Result": "LabResult"})
rx = T["Prescriptions"][["AppointmentID", "Medication"]]
adm = T["Admissions"][["AppointmentID", "LengthOfStayDays"]]
rf = T["Referrals"][["AppointmentID", "ReferredFacilityID"]]
doc_small = doc[["StaffID", "Specialty", "YearsExperience"]].rename(columns={"StaffID": "DoctorID", "Specialty": "DoctorSpecialty", "YearsExperience": "DoctorYearsExp"})
fin_cols = F[["AppointmentID", "BillingID", "Amount", "Status", "AmountPaid", "Method", "Outstanding", "PaidRatio", "ClaimID", "Provider", "ClaimAmount", "ClaimStatus", "ClaimRatio", "HasClaim"]].rename(
    columns={"Amount": "BillAmount", "Status": "BillStatus", "Method": "PayMethod", "Provider": "ClaimProvider"})
fbk = fb[["AppointmentID", "Rating", "Comment"]].rename(columns={"Comment": "FeedbackComment"})
dxu = D["Diagnoses_csv"][["AppointmentID", "Diagnosis", "Severity"]]

E = (ap.merge(PATIENTS, on="PatientID", how="left").merge(doc_small, on="DoctorID", how="left")
     .merge(dx[["AppointmentID", "Diagnosis", "Severity"]], on="AppointmentID", how="left")
     .merge(lab, on="AppointmentID", how="left").merge(rx, on="AppointmentID", how="left")
     .merge(adm, on="AppointmentID", how="left").merge(rf, on="AppointmentID", how="left")
     .merge(fin_cols, on="AppointmentID", how="left").merge(fbk, on="AppointmentID", how="left"))
assert len(E) == len(ap) and E.AppointmentID.is_unique
E["Admitted"] = E.LengthOfStayDays.notna().astype(int)
E["LabDone"] = E.LabTest.notna().astype(int)
E["LabAbnormal"] = (E.LabResult == "Abnormal").astype(int)
E["PrescriptionIssued"] = E.Medication.notna().astype(int)
E["Referred"] = E.ReferredFacilityID.notna().astype(int)
E["HasBill"] = E.BillingID.notna().astype(int)
E["HasFeedback"] = E.Rating.notna().astype(int)
E["DayOfWeek"] = E.Date.dt.day_name()
E["MonthNum"] = E.Date.dt.month
E["Quarter"] = E.Date.dt.quarter
E = E.drop(columns=["Dept_Facility", "Doc_Facility", "Doc_DeptID", "Doc_DeptName"], errors="ignore")
cov = {k: int(E[k].sum()) for k in ["Admitted", "LabDone", "PrescriptionIssued", "Referred", "HasBill", "HasFeedback"]}
R["encounter_coverage"] = cov
print("Encounter master:", E.shape, "| coverage of linked records (of 5,000 appointments):", cov)

FIN = (F.merge(dxu, on="AppointmentID", how="left").merge(fbk, on="AppointmentID", how="left")
       .merge(ap[["AppointmentID", "FacilityID", "DepartmentID", "DepartmentName", "PatientID", "VisitType", "Date"]], on="AppointmentID", how="left"))
FIN = FIN.rename(columns={"Amount": "BillAmount", "Status": "BillStatus", "Method": "PayMethod", "Provider": "ClaimProvider"})
FIN["InCohort"] = FIN.FacilityID.notna().astype(int)
FIN["SeverityScore"] = FIN.Severity.map({"Mild": 1, "Moderate": 2, "Severe": 3})
print("Financial/feedback universe:", FIN.shape, "| bills linkable to an appointment:", int(FIN.InCohort.sum()))
R["n_encounters"], R["n_fin"], R["n_patients"] = len(E), len(FIN), len(PATIENTS)
print("Integrated-dataset checks → encounters unique:", E.AppointmentID.is_unique, "| fin unique:", FIN.BillingID.is_unique)
E.head(3).T

save(globals(), __file__)
print("Done:", os.path.basename(__file__))
