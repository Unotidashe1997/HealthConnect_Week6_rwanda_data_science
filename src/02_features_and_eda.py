"""
HealthConnect Rwanda - Week 6 (Data Science track)
Part 2 - Feature engineering and exploratory data analysis (figures 1-17)

Run from the repository root:  python src/02_features_and_eda.py
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
# ## 2. Feature engineering
# Ten engineered variables (all required by the brief) plus supporting helpers. Definitions are explicit so management
# can reproduce them:
#
# | Feature | Definition |
# |---|---|
# | Patient Age Group | 0–4, 5–14, 15–24, 25–34, 35–49, 50–64, 65+ |
# | High-Risk Patient Indicator | Rule-based score: age ≥65 (+2), age 50–64 (+1), any chronic disease (+1), BMI ≥30 (+1), BMI <18.5 (+1); **high risk if score ≥ 3** |
# | Length-of-Stay Category | Not admitted · Short (1–3 d) · Medium (4–7 d) · Long (8–10 d) |
# | Frequent Visitor Indicator | Patient with ≥ 2 appointments in the 2026 appointment record |
# | Insurance Coverage Category | From claim ÷ bill: No claim · Low (<60 %) · Medium (60–85 %) · High (≥85 %) |
# | Revenue Category | Bill amount terciles: Low / Medium / High |
# | Disease Severity Score | Mild = 1, Moderate = 2, Severe = 3 |
# | Facility Performance Score | 0–100 composite: satisfaction 35 %, fee collection 30 %, claim approval 15 %, clinician utilisation 20 % |
# | Patient Satisfaction Category | Rating 1–2 Dissatisfied · 3 Neutral · 4–5 Satisfied |
# | Department Workload Index | Appointments per clinician (doctors + nurses) ÷ system average × 100 |

# %%
AGE_BINS = [0, 5, 15, 25, 35, 50, 65, 200]
AGE_LAB = ["0–4", "5–14", "15–24", "25–34", "35–49", "50–64", "65+"]
age_group = lambda s: pd.cut(s, AGE_BINS, labels=AGE_LAB, right=False)

def risk_score(df):
    return ((df.Age >= 65) * 2 + ((df.Age >= 50) & (df.Age < 65)) * 1 + df.ChronicFlag + (df.BMI >= 30) * 1 + (df.BMI < 18.5) * 1)

for df in (PATIENTS, E):
    df["AgeGroup"] = age_group(df.Age)
    df["RiskScore"] = risk_score(df)
    df["HighRisk"] = (df.RiskScore >= 3).astype(int)
    df["BMICategory"] = pd.cut(df.BMI, [0, 18.5, 25, 30, 100], labels=["Underweight", "Normal", "Overweight", "Obese"], right=False)

# frequent visitor (appointment record) ------------------------------------------------------------------
visits = E.PatientID.value_counts()
E["PatientVisits"] = E.PatientID.map(visits)
E["FrequentVisitor"] = (E.PatientVisits >= 2).astype(int)
PATIENTS["Visits2026"] = PATIENTS.PatientID.map(visits).fillna(0).astype(int)
PATIENTS["FrequentVisitor"] = (PATIENTS.Visits2026 >= 2).astype(int)

# length of stay category --------------------------------------------------------------------------------
E["LOSCategory"] = pd.cut(E.LengthOfStayDays, [0, 3, 7, 10], labels=["Short (1–3 d)", "Medium (4–7 d)", "Long (8–10 d)"]).astype(object)
E["LOSCategory"] = E.LOSCategory.fillna("Not admitted")

# insurance coverage category (claim ÷ bill) -------------------------------------------------------------
def cov_cat(r):
    if pd.isna(r): return "No claim"
    return "Low (<60%)" if r < .6 else ("Medium (60–85%)" if r < .85 else "High (≥85%)")
for df in (E, FIN):
    df["CoverageCategory"] = (df.ClaimRatio if "ClaimRatio" in df else df.ClaimAmount / df.BillAmount).map(cov_cat)

# revenue category ----------------------------------------------------------------------------------------
q1, q2 = FIN.BillAmount.quantile([1 / 3, 2 / 3])
rev_cat = lambda s: pd.cut(s, [-np.inf, q1, q2, np.inf], labels=["Low", "Medium", "High"])
FIN["RevenueCategory"] = rev_cat(FIN.BillAmount); E["RevenueCategory"] = rev_cat(E.BillAmount)
R["revenue_tercile_cuts"] = [float(q1), float(q2)]

# severity score & satisfaction category ----------------------------------------------------------------
sev_map = {"Mild": 1, "Moderate": 2, "Severe": 3}
E["SeverityScore"] = E.Severity.map(sev_map)
sat_cat = lambda s: pd.cut(s, [0, 2, 3, 5], labels=["Dissatisfied", "Neutral", "Satisfied"])
E["SatisfactionCategory"] = sat_cat(E.Rating); FIN["SatisfactionCategory"] = sat_cat(FIN.Rating)

# staff capacity tables ------------------------------------------------------------------------------------
STAFF["Clinician"] = STAFF.Cadre.isin(["Doctors", "Nurses"]).astype(int)
staff_fac = STAFF.pivot_table(index="FacilityID", columns="Cadre", values="StaffID", aggfunc="count", fill_value=0)
staff_fac["TotalStaff"] = staff_fac.sum(axis=1)
clin_dept = STAFF[STAFF.Clinician == 1].groupby("DepartmentID").size().rename("Clinicians")       # attribute by DepartmentID
beds_per_fac = (D["Beds"].merge(D["Rooms"], on="RoomID").groupby("FacilityID").size().rename("Beds"))
rooms_per_fac = D["Rooms"].groupby("FacilityID").size().rename("Rooms")
equip_per_fac = D["Medical_Equipment"].groupby("FacilityID").size().rename("Equipment")
n_dept_fac = dept.groupby("FacilityID").size().rename("Departments")

# department workload index (department-ID level and department-type level) ---------------------------
DEPT = (dept.set_index("DepartmentID").join(E.groupby("DepartmentID").agg(Appointments=("AppointmentID", "size"), Emergency=("VisitType", lambda s: (s == "Emergency").mean()),
        AdmitRate=("Admitted", "mean"), LabPerAppt=("LabDone", "mean"), RxPerAppt=("PrescriptionIssued", "mean"), MeanSeverity=("SeverityScore", "mean"),
        MeanBill=("BillAmount", "mean"), MeanRating=("Rating", "mean")))
        .join(clin_dept))
DEPT[["Appointments", "Clinicians"]] = DEPT[["Appointments", "Clinicians"]].fillna(0)
DEPT["ApptPerClinician"] = DEPT.Appointments / DEPT.Clinicians.replace(0, np.nan)
DEPT["WorkloadIndex"] = 100 * DEPT.ApptPerClinician / DEPT.ApptPerClinician.mean()
DEPTT = (DEPT.groupby("DepartmentName").agg(Appointments=("Appointments", "sum"), Clinicians=("Clinicians", "sum"), n_depts=("Appointments", "size")))
DEPTT["ApptPerClinician"] = DEPTT.Appointments / DEPTT.Clinicians
DEPTT["WorkloadIndex"] = 100 * DEPTT.ApptPerClinician / DEPTT.ApptPerClinician.mean()
E["DeptWorkloadIndex"] = E.DepartmentID.map(DEPT.WorkloadIndex)
E["DeptTypeWorkloadIndex"] = E.DepartmentName.map(DEPTT.WorkloadIndex)
print("Features created. High-risk patients:", pct(PATIENTS.HighRisk.mean()), "| frequent visitors (of patients seen):", pct(E.drop_duplicates('PatientID').FrequentVisitor.mean()))
R["highrisk_pct"] = float(PATIENTS.HighRisk.mean())
DEPTT.round(2)

# %%
# facility table: operations, clinical, financial and experience measures (used by Performance Score, EDA and clustering)
g = E.groupby("FacilityID")
K = pd.DataFrame({
    "Appointments": g.size(), "UniquePatients": g.PatientID.nunique(),
    "EmergencyShare": g.VisitType.apply(lambda s: (s == "Emergency").mean()), "AdmitRate": g.Admitted.mean(),
    "MeanLOS": g.LengthOfStayDays.mean(), "ReferralRate": g.Referred.mean(), "MeanSeverity": g.SeverityScore.mean(),
    "LabPerAppt": g.LabDone.mean(), "MeanBill": g.BillAmount.mean(), "BilledTotal": g.BillAmount.sum(), "PaidTotal": g.AmountPaid.sum(),
    "ClaimApproval": g.apply(lambda d: (d.ClaimStatus == "Approved").sum() / d.ClaimStatus.notna().sum()),
    "MeanRating": g.Rating.mean(), "RatedN": g.Rating.count(),
}).join(staff_fac[["Doctors", "Nurses", "TotalStaff"]]).join(beds_per_fac).join(rooms_per_fac).join(equip_per_fac).join(n_dept_fac)
K["CollectionRate"] = K.PaidTotal / K.BilledTotal
K["ApptPerDoctor"] = K.Appointments / K.Doctors
K["ApptPerClinician"] = K.Appointments / (K.Doctors + K.Nurses)
K["StaffPer1000Pts"] = 1000 * K.TotalStaff / K.UniquePatients
K["ApptPerBed"] = K.Appointments / K.Beds
z = lambda s: (s - s.mean()) / s.std(ddof=0)
raw = 0.35 * z(K.MeanRating) + 0.30 * z(K.CollectionRate) + 0.15 * z(K.ClaimApproval) + 0.20 * z(K.ApptPerClinician)
K["PerformanceScore"] = 100 * (raw - raw.min()) / (raw.max() - raw.min())
K["PerfRank"] = K.PerformanceScore.rank(ascending=False).astype(int)
E["FacilityPerformanceScore"] = E.FacilityID.map(K.PerformanceScore)
K.round(3).sort_values("PerfRank")[["Appointments", "UniquePatients", "MeanRating", "CollectionRate", "ClaimApproval", "ApptPerClinician", "PerformanceScore", "PerfRank"]].head(20)

# %% [markdown]
# ## 3. Exploratory data analysis
# Chart conventions: fixed categorical colour order, one axis per chart, muted grid, direct title/axis labels, and a
# plain-language interpretation under every figure (generated from the plotted numbers).
# ### 3.1 Patient analysis (20,000 registered patients)

# %%
n_pat = len(PATIENTS)
gender = PATIENTS.Gender.value_counts()
bg = PATIENTS.BloodGroup.value_counts()
chron = PATIENTS.ChronicCondition.value_counts()
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0, 0].bar(gender.index, gender.values, color=[C[0], C[1]], width=.55)
for i, v in enumerate(gender.values): ax[0, 0].text(i, v + 100, f"{v:,}\n({pct(v / n_pat)})", ha="center", fontsize=9)
ax[0, 0].set(title="Gender distribution", ylabel="Patients", ylim=(0, gender.max() * 1.18))
for gname, col in zip(["Male", "Female"], [C[0], C[1]]):
    sns.kdeplot(PATIENTS.loc[PATIENTS.Gender == gname, "Age"], ax=ax[0, 1], color=col, label=gname, fill=True, alpha=.15, bw_adjust=.7, clip=(0, 90))
ax[0, 1].axvline(PATIENTS.Age.mean(), color=INK2, ls="--", lw=1); ax[0, 1].text(PATIENTS.Age.mean() + 1, ax[0, 1].get_ylim()[1] * .9, f"mean {PATIENTS.Age.mean():.1f}", fontsize=8.5, color=INK2)
ax[0, 1].set(title="Age distribution (density) by gender", xlabel="Age (years)", ylabel="Density"); ax[0, 1].legend()
ax[1, 0].bar(bg.index, bg.values, color=C[0], width=.6)
ax[1, 0].set(title="Blood group distribution", ylabel="Patients", ylim=(0, bg.max() * 1.12))
for i, v in enumerate(bg.values): ax[1, 0].text(i, v + 40, pct(v / n_pat), ha="center", fontsize=8.5)
cc = chron.reindex(["None", "Asthma", "Hypertension", "Diabetes"])
ax[1, 1].bar(cc.index, cc.values, color=[MUTED, C[0], C[1], C[2]], width=.6)
for i, v in enumerate(cc.values): ax[1, 1].text(i, v + 80, pct(v / n_pat), ha="center", fontsize=9)
ax[1, 1].set(title="Chronic-condition prevalence", ylabel="Patients", ylim=(0, cc.max() * 1.15))
fig.suptitle("Who are HealthConnect's patients?", fontsize=13.5, fontweight="semibold", y=1.0)
fig.tight_layout()
chi_g = stats.chisquare(bg.values)
R["gender_split"] = {k: int(v) for k, v in gender.items()}; R["mean_age"] = float(PATIENTS.Age.mean()); R["chronic_prev"] = float(PATIENTS.ChronicFlag.mean())
R["blood_uniform_p"] = float(chi_g.pvalue)
save_fig(fig, "f01_patient_profile",
         f"The register is balanced by gender ({pct(gender['Male'] / n_pat)} male) and spans all ages (mean {PATIENTS.Age.mean():.1f} y). "
         f"{pct(PATIENTS.ChronicFlag.mean())} of patients carry a chronic condition, split almost equally across asthma, hypertension and diabetes. "
         f"Blood groups are near-uniform (χ² test for equal shares {fmt_p(chi_g.pvalue)}), which is unusual for a real population where O and A dominate — another sign the data are simulated.")

# %%
pyr = PATIENTS.assign(B=pd.cut(PATIENTS.Age, range(0, 95, 5), right=False)).groupby(["B", "Gender"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(8.5, 6.5))
y = np.arange(len(pyr))
ax.barh(y, -pyr["Male"], color=C[0], height=.8, label="Male"); ax.barh(y, pyr["Female"], color=C[1], height=.8, label="Female")
ax.set_yticks(y); ax.set_yticklabels([f"{int(i.left)}–{int(i.right) - 1}" for i in pyr.index])
mx = max(pyr.max()) * 1.1; ax.set_xlim(-mx, mx); ax.set_xticks(ax.get_xticks()); ax.set_xticklabels([f"{abs(int(t)):,}" for t in ax.get_xticks()])
ax.set(title="Age pyramid of registered patients", xlabel="Patients", ylabel="Age band (years)"); ax.legend(loc="lower right"); ax.axvline(0, color=INK2, lw=.8)
under15 = (PATIENTS.Age < 15).mean(); over65 = (PATIENTS.Age >= 65).mean()
msh = 100 * pyr.Male / (pyr.Male + pyr.Female)
R["under15"], R["over65"] = float(under15), float(over65)
save_fig(fig, "f02_age_pyramid", f"The pyramid is rectangular rather than the narrow-topped shape of a real East-African population: children under 15 are {pct(under15)} and adults 65+ are {pct(over65)} of patients. "
         f"Male share stays between {msh.min():.0f}% and {msh.max():.0f}% in every five-year band, so neither sex is concentrated in any age group; this evenness is another indicator of synthetic data.")

# %%
pg = PATIENTS.groupby("AgeGroup", observed=True).agg(n=("PatientID", "size"), chronic=("ChronicFlag", "mean"), highrisk=("HighRisk", "mean"), bmi=("BMI", "mean"))
fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
ax[0].bar(pg.index.astype(str), 100 * pg.chronic, color=C[0], width=.65); ax[0].set(title="Chronic-disease prevalence by age group", xlabel="Age group", ylabel="% of patients with a chronic condition", ylim=(0, 100))
ax[0].axhline(100 * PATIENTS.ChronicFlag.mean(), color=INK2, ls="--", lw=1)
for i, v in enumerate(pg.chronic): ax[0].text(i, 100 * v + 1.5, f"{100 * v:.0f}%", ha="center", fontsize=8.5)
ct = pd.crosstab(PATIENTS.AgeGroup, PATIENTS.ChronicCondition, normalize="index")[["Asthma", "Hypertension", "Diabetes"]] * 100
bottom = np.zeros(len(ct))
for i, c in enumerate(ct.columns):
    ax[1].bar(ct.index.astype(str), ct[c], bottom=bottom, color=C[i], width=.65, label=c, edgecolor=SURF, linewidth=1.2); bottom += ct[c].values
ax[1].set(title="Chronic condition mix by age group", xlabel="Age group", ylabel="% of patients"); ax[1].legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, -.22))
ax[2].bar(pg.index.astype(str), 100 * pg.highrisk, color=C[1], width=.65); ax[2].set(title="High-risk patients (score ≥ 3) by age group", xlabel="Age group", ylabel="% high risk")
for i, v in enumerate(pg.highrisk): ax[2].text(i, 100 * v + 1, f"{100 * v:.0f}%", ha="center", fontsize=8.5)
fig.tight_layout()
chi_age_chr = stats.chi2_contingency(pd.crosstab(PATIENTS.AgeGroup, PATIENTS.ChronicFlag))
chi_sex_chr = stats.chi2_contingency(pd.crosstab(PATIENTS.Gender, PATIENTS.ChronicFlag))
R["chronic_age_p"], R["chronic_sex_p"] = float(chi_age_chr[1]), float(chi_sex_chr[1])
save_fig(fig, "f03_chronic_by_age",
         f"Chronic-disease prevalence is flat across age groups (range {100 * pg.chronic.min():.0f}–{100 * pg.chronic.max():.0f}%; χ² {fmt_p(chi_age_chr[1])}) and between sexes ({fmt_p(chi_sex_chr[1])}). "
         f"The risk score therefore rises with age almost entirely through the age points themselves: {pct(pg.highrisk.iloc[-1], 0)} of patients 65+ are high-risk versus {pct(pg.highrisk.iloc[0], 0)} of under-5s. "
         "In real populations chronic disease rises steeply with age, so the absence of a gradient is a data-quality limitation, not a clinical finding.")

# %%
ins = PATIENTS.InsuranceID.value_counts().sort_index()
chi_ins = stats.chisquare(ins.values)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
ax[0].bar(ins.index.str.replace("INS", ""), ins.values, color=C[0], width=.7)
ax[0].axhline(ins.mean(), color=INK2, ls="--", lw=1); ax[0].text(len(ins) - .5, ins.mean() + 12, f"mean {ins.mean():,.0f}", ha="right", fontsize=8.5, color=INK2)
ax[0].set(title="Patients by insurance code (INS001–INS015)", xlabel="Insurance code", ylabel="Patients", ylim=(0, ins.max() * 1.1))
cl = FIN.assign(Cov=FIN.CoverageCategory).Cov.value_counts().reindex(["No claim", "Low (<60%)", "Medium (60–85%)", "High (≥85%)"])
ax[1].barh(cl.index[::-1], cl.values[::-1], color=[C[0], C[2], C[3], MUTED], height=.6)
for i, v in enumerate(cl.values[::-1]): ax[1].text(v + 400, i, f"{v:,} ({pct(v / len(FIN))})", va="center", fontsize=8.5)
ax[1].set(title="Claim coverage of bills (60,000)", xlabel="Bills", xlim=(0, cl.max() * 1.35))
fig.tight_layout()
R["insurance_cov_all"] = 1.0; R["claim_rate"] = float(FIN.HasClaim.mean())
save_fig(fig, "f04_insurance",
         f"Every patient holds one of 15 insurance codes in near-equal numbers (χ² for equal shares {fmt_p(chi_ins.pvalue)}), so the insurance coverage rate is effectively 100% and no uninsured group exists to compare. "
         f"Insurance matters instead at billing: {pct(FIN.HasClaim.mean())} of bills carry a claim, and the claim usually covers only part of the bill (median coverage {pct(FIN.ClaimRatio.median(), 0)}).")

# %%
prov = PATIENTS.groupby("ProvinceName").agg(Patients=("PatientID", "size"), Chronic=("ChronicFlag", "mean"), HighRisk=("HighRisk", "mean"), MeanAge=("Age", "mean")).sort_values("Patients", ascending=False)
dist = PATIENTS.groupby("DistrictName").size().sort_values(ascending=False)
centroid = {"Kigali City": (30.06, -1.95), "Eastern": (30.55, -1.95), "Northern": (29.85, -1.55), "Southern": (29.75, -2.45), "Western": (29.30, -2.05)}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1, 1.4, 1.1]})
ax[0].barh(prov.index[::-1], prov.Patients[::-1], color=C[0], height=.6)
for i, v in enumerate(prov.Patients[::-1]): ax[0].text(v + 60, i, f"{v:,}", va="center", fontsize=8.5)
ax[0].set(title="Patients by province", xlabel="Patients", xlim=(0, prov.Patients.max() * 1.18))
ax[1].barh(dist.index[:15][::-1], dist.values[:15][::-1], color=C[0], height=.65); ax[1].set(title="Top 15 districts by patients (of 30)", xlabel="Patients")
ax[1].axvline(dist.mean(), color=INK2, ls="--", lw=1)
xs = [centroid[p][0] for p in prov.index]; ys = [centroid[p][1] for p in prov.index]
sc = ax[2].scatter(xs, ys, s=prov.Patients / 6, c=100 * prov.Chronic, cmap=SEQ, edgecolor=INK2, linewidth=1, vmin=70, vmax=80, zorder=3)
for p, x_, y_ in zip(prov.index, xs, ys): ax[2].annotate(f"{p}\n{prov.loc[p, 'Patients']:,}", (x_, y_), textcoords="offset points", xytext=(0, -34), ha="center", fontsize=8)
ax[2].set(title="Schematic province map (bubble = patients, colour = % chronic)", xlim=(28.9, 30.9), ylim=(-2.85, -1.2), xlabel="Longitude (approx. centroid)", ylabel="Latitude (approx. centroid)"); ax[2].grid(False)
cb = fig.colorbar(sc, ax=ax[2], fraction=.05, pad=.02); cb.set_label("% chronic")
fig.tight_layout()
chi_prov = stats.chisquare(prov.Patients.values)
R["province_patients"] = prov.Patients.to_dict(); R["top_province"] = prov.index[0]
save_fig(fig, "f05_geography",
         f"{prov.index[0]} has the most registered patients ({prov.Patients.iloc[0]:,}, {pct(prov.Patients.iloc[0] / n_pat)}) and {prov.index[-1]} the fewest ({prov.Patients.iloc[-1]:,}); the spread is modest (χ² for equal shares {fmt_p(chi_prov.pvalue)}). "
         "Every province contains exactly six districts, so near-equal shares are structural rather than a demand signal. "
         "The map is schematic: the data contain no coordinates or boundary files, so bubbles sit at approximate province centroids.")

# %%
first = E.groupby("PatientID").Date.min().dt.to_period("M").dt.to_timestamp()
newp = first.value_counts().sort_index()
cum = newp.cumsum()
vol = E.groupby("Month").size()
ret = vol - newp.reindex(vol.index).fillna(0)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
ax[0].plot(cum.index, cum.values, color=C[0], lw=2, marker="o", ms=4); ax[0].set(title="Cumulative unique patients seen (growth proxy)", xlabel="Month of first appointment (2026)", ylabel="Unique patients")
ax[0].text(cum.index[-1], cum.values[-1], f" {cum.values[-1]:,}", va="center", fontsize=9)
w = 8
ax[1].bar(newp.index - pd.Timedelta(days=4), newp.values, width=w, color=C[0], label="New patients"); ax[1].bar(ret.index + pd.Timedelta(days=4), ret.values, width=w, color=C[1], label="Returning patients")
ax[1].set(title="New vs returning patients per month", xlabel="Month (2026)", ylabel="Patients"); ax[1].legend()
fig.tight_layout()
R["unique_patients_seen"] = int(len(first)); R["returning_patients"] = int((visits >= 2).sum()); R["patients_seen_pct"] = float(len(first) / n_pat)
save_fig(fig, "f06_patient_growth",
         f"The 5,000 appointments involve {len(first):,} distinct patients ({pct(len(first) / n_pat)} of the register); {int((visits >= 2).sum())} patients visited more than once. "
         f"About {len(first) / 12:.0f} new patients appear per month with no visible acceleration, which is what an unrelated random draw from the register would produce. Registration dates are not recorded, so first-appointment month is only a proxy for patient growth.")

# %% [markdown]
# ### 3.2 Clinical analysis

# %%
dxu_all = D["Diagnoses_csv"]
dfreq = dxu_all.Diagnosis.value_counts()
sev_ct = pd.crosstab(dxu_all.Diagnosis, dxu_all.Severity, normalize="index")[["Mild", "Moderate", "Severe"]] * 100
sev_ct = sev_ct.loc[dfreq.index]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.2, 1.3, 1]})
ax[0].barh(dfreq.index[::-1], dfreq.values[::-1], color=C[0], height=.6)
for i, v in enumerate(dfreq.values[::-1]): ax[0].text(v + 150, i, f"{v:,} ({pct(v / len(dxu_all))})", va="center", fontsize=8.5)
ax[0].set(title="Diagnosis frequency (60,000 diagnoses)", xlabel="Diagnoses", xlim=(0, dfreq.max() * 1.4))
left = np.zeros(len(sev_ct))
for i, s in enumerate(["Mild", "Moderate", "Severe"]):
    ax[1].barh(sev_ct.index[::-1], sev_ct[s][::-1], left=left, color=[C[2], C[3], C[7]][i], label=s, height=.6, edgecolor=SURF, linewidth=1.2); left += sev_ct[s][::-1].values
ax[1].set(title="Severity mix within each diagnosis", xlabel="% of diagnoses"); ax[1].legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, -.18)); ax[1].set_xlim(0, 100)
sv = dxu_all.Severity.value_counts().reindex(["Mild", "Moderate", "Severe"])
ax[2].bar(sv.index, sv.values, color=[C[2], C[3], C[7]], width=.6)
for i, v in enumerate(sv.values): ax[2].text(i, v + 150, pct(v / len(dxu_all)), ha="center", fontsize=9)
ax[2].set(title="Overall disease severity", ylabel="Diagnoses", ylim=(0, sv.max() * 1.12))
fig.tight_layout()
chi_dx_sev = stats.chi2_contingency(pd.crosstab(dxu_all.Diagnosis, dxu_all.Severity))
R["dx_freq"] = dfreq.to_dict(); R["severity_mix"] = (sv / len(dxu_all)).to_dict(); R["dx_sev_p"] = float(chi_dx_sev[1])
save_fig(fig, "f07_diagnoses",
         f"Five diagnoses share the caseload almost equally — {dfreq.index[0]} leads with {pct(dfreq.iloc[0] / len(dxu_all))} and {dfreq.index[-1]} trails with {pct(dfreq.iloc[-1] / len(dxu_all))}. "
         f"Severity is split roughly in thirds (severe {pct(sv['Severe'] / len(dxu_all))}), and the severity mix is the same for every disease (χ² {fmt_p(chi_dx_sev[1])}). "
         "A one-third severe rate is far higher than in usual primary-care case-mix; management should confirm how severity is coded.")

# %%
labs_t = T["Laboratory_Tests"]; rx_t = T["Prescriptions"]
lab_use = labs_t.Test.value_counts(); lab_abn = labs_t.groupby("Test").Result.apply(lambda s: (s == "Abnormal").mean()).reindex(lab_use.index)
rx_use = rx_t.Medication.value_counts()
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
ax[0].bar(lab_use.index, lab_use.values, color=C[0], width=.6)
for i, v in enumerate(lab_use.values): ax[0].text(i, v + 15, f"{v:,}", ha="center", fontsize=8.5)
ax[0].set(title="Laboratory test volume", ylabel="Tests", ylim=(0, lab_use.max() * 1.12))
ax[1].bar(lab_abn.index, 100 * lab_abn.values, color=C[1], width=.6); ax[1].axhline(100 * (labs_t.Result == "Abnormal").mean(), color=INK2, ls="--", lw=1)
for i, v in enumerate(lab_abn.values): ax[1].text(i, 100 * v + 1, f"{100 * v:.0f}%", ha="center", fontsize=8.5)
ax[1].set(title="Abnormal-result rate by test", ylabel="% abnormal", ylim=(0, 70))
ax[2].barh(rx_use.index[::-1], rx_use.values[::-1], color=C[2], height=.6)
for i, v in enumerate(rx_use.values[::-1]): ax[2].text(v + 12, i, f"{v:,}", va="center", fontsize=8.5)
ax[2].set(title="Prescription frequency by medication", xlabel="Prescriptions", xlim=(0, rx_use.max() * 1.15))
fig.tight_layout()
# does the medication fit the diagnosis?  (clinical-logic check)
mx_ct = pd.crosstab(E.Diagnosis, E.Medication)
chi_med = stats.chi2_contingency(mx_ct)
conc_map = {"Diabetes": "Metformin", "Gastritis": "Omeprazole", "Hypertension": "Amlodipine", "Upper Respiratory Infection": "Amoxicillin", "Malaria": "Artemether-Lumefantrine"}
_m = E[E.Medication.notna()]
rx_conc = float((_m.Medication == _m.Diagnosis.map(conc_map)).mean())
lab_dx = stats.chi2_contingency(pd.crosstab(E.Diagnosis[E.LabTest.notna()], E.LabTest[E.LabTest.notna()]))
lab_match = float((E[E.LabTest.notna()].LabTest == E[E.LabTest.notna()].Diagnosis.map({"Malaria": "Malaria", "Diabetes": "Glucose"})).sum() / E.Diagnosis[E.LabTest.notna()].isin(["Malaria", "Diabetes"]).sum())
R["rx_concordance"], R["lab_dx_p"] = rx_conc, float(lab_dx[1])
R["lab_volume"] = lab_use.to_dict(); R["rx_volume"] = rx_use.to_dict(); R["abnormal_rate"] = float((labs_t.Result == "Abnormal").mean()); R["med_dx_p"] = float(chi_med[1])
save_fig(fig, "f08_labs_rx",
         f"Lab volume is evenly spread over four tests (CBC, Glucose, Malaria, Urinalysis; {pct(lab_use.min() / lab_use.sum(), 0)}–{pct(lab_use.max() / lab_use.sum(), 0)} each) and {pct((labs_t.Result == 'Abnormal').mean(), 0)} of results are abnormal (range {100 * lab_abn.min():.0f}–{100 * lab_abn.max():.0f}% by test). "
         f"Prescribing follows diagnosis exactly — {pct(rx_conc, 0)} of prescriptions match the standard drug for the diagnosis (Metformin for diabetes, Amlodipine for hypertension, Omeprazole for gastritis, Amoxicillin for URI, Artemether-Lumefantrine for malaria) — so {rx_use.index[0]} leads only because it follows the diagnosis mix. "
         f"Laboratory ordering, by contrast, is unrelated to diagnosis (χ² {fmt_p(lab_dx[1])}): malaria patients are not more likely to receive a malaria test, which management should review as a possible test-ordering or data-capture gap.")

# %%
adm_m = E.groupby("Month").agg(Admissions=("Admitted", "sum"), Appts=("AppointmentID", "size"), Referrals=("Referred", "sum"))
adm_m["AdmitRate"] = 100 * adm_m.Admissions / adm_m.Appts; adm_m["RefRate"] = 100 * adm_m.Referrals / adm_m.Appts
adm_all = T["Admissions"]
ref_dest = T["Referrals"].ReferredFacilityID.value_counts()
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
ax[0].plot(adm_m.index, adm_m.AdmitRate, color=C[0], lw=2, marker="o", ms=4, label="Admission rate")
ax[0].plot(adm_m.index, adm_m.RefRate, color=C[1], lw=2, marker="s", ms=4, label="Referral rate")
ax[0].axhline(100 * E.Admitted.mean(), color=C[0], ls=":", lw=1); ax[0].axhline(100 * E.Referred.mean(), color=C[1], ls=":", lw=1)
ax[0].set(title="Monthly admission and referral rate", xlabel="Month (2026)", ylabel="% of appointments", ylim=(0, 20)); ax[0].legend(loc="upper right", ncol=2)
ax[1].hist(adm_all.LengthOfStayDays, bins=np.arange(.5, 11.5, 1), color=C[0], edgecolor=SURF, rwidth=.9); ax[1].axvline(adm_all.LengthOfStayDays.mean(), color=INK2, ls="--", lw=1)
ax[1].text(adm_all.LengthOfStayDays.mean() + .15, ax[1].get_ylim()[1] * .92, f"mean {adm_all.LengthOfStayDays.mean():.1f} d", fontsize=8.5, color=INK2)
ax[1].set(title="Length of stay (5,000 admissions)", xlabel="Days", ylabel="Admissions"); ax[1].set_xticks(range(1, 11))
ax[2].barh(ref_dest.index[::-1][:12], ref_dest.values[::-1][:12], color=C[2], height=.65); ax[2].set(title="Top referral destinations (4,708 referrals)", xlabel="Referrals")
fig.tight_layout()
trend_adm = stats.spearmanr(np.arange(12), adm_m.AdmitRate)
los_u = stats.chisquare(adm_all.LengthOfStayDays.value_counts().sort_index().values); ref_u = stats.chisquare(ref_dest.values)
R["admit_rate"] = float(E.Admitted.mean()); R["mean_los"] = float(adm_all.LengthOfStayDays.mean()); R["referral_rate"] = float(E.Referred.mean())
R["n_admissions_all"] = int(len(adm_all)); R["admit_trend_rho"], R["admit_trend_p"] = float(trend_adm[0]), float(trend_adm[1])
save_fig(fig, "f09_admissions_referrals",
         f"{pct(E.Admitted.mean())} of appointments end in admission (586 of 5,000) with a mean stay of {adm_all.LengthOfStayDays.mean():.1f} days; stay lengths 1–10 days are equally likely (χ² {fmt_p(los_u.pvalue)}). "
         f"Monthly admission rates bounce between {adm_m.AdmitRate.min():.1f}% and {adm_m.AdmitRate.max():.1f}% with no trend (Spearman ρ = {trend_adm[0]:.2f}, {fmt_p(trend_adm[1])}). "
         f"Referrals touch {pct(E.Referred.mean())} of appointments; destinations range {ref_dest.min()}–{ref_dest.max()} referrals per facility and are statistically even (χ² {fmt_p(ref_u.pvalue)}), so no facility acts as a referral hub.")

# %% [markdown]
# ### 3.3 Operational analysis

# %%
doc_load = E.groupby("DoctorID").size().reindex(doc.StaffID, fill_value=0)
nurse_fac = D["Nurses"].groupby("FacilityID").size()
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
ax[0].hist(doc_load, bins=np.arange(-.5, doc_load.max() + 1.5, 1), color=C[0], edgecolor=SURF, rwidth=.9)
ax[0].axvline(doc_load.mean(), color=INK2, ls="--", lw=1); ax[0].text(doc_load.mean() + .3, ax[0].get_ylim()[1] * .92, f"mean {doc_load.mean():.1f}", fontsize=8.5, color=INK2)
ax[0].set(title="Appointments per doctor (350 doctors)", xlabel="Appointments in 2026", ylabel="Doctors")
topd = doc_load.sort_values(ascending=False).head(12)
ax[1].barh(topd.index[::-1], topd.values[::-1], color=C[1], height=.65); ax[1].set(title="12 busiest doctors", xlabel="Appointments")
ax[2].scatter(K.Nurses, K.Appointments / K.Nurses, s=55, color=C[0], edgecolor=SURF, zorder=3)
for f_, r_ in K.iterrows(): ax[2].annotate(f_[-2:], (r_.Nurses, r_.Appointments / r_.Nurses), textcoords="offset points", xytext=(4, 3), fontsize=7.5, color=INK2)
ax[2].set(title="Nurse workload by facility", xlabel="Nurses at facility", ylabel="Appointments per nurse")
fig.tight_layout()
cv = doc_load.std() / doc_load.mean()
R["doc_load_mean"], R["doc_load_max"], R["doc_load_cv"] = float(doc_load.mean()), int(doc_load.max()), float(cv)
exp_pois = stats.poisson(doc_load.mean())
disp = doc_load.var() / doc_load.mean()
R["doc_dispersion"] = float(disp)
save_fig(fig, "f10_staff_workload",
         f"Doctors handle {doc_load.mean():.1f} appointments each on average (range {doc_load.min()}–{doc_load.max()}); the variance-to-mean ratio is {disp:.2f}, i.e. almost exactly what random assignment (Poisson ≈ 1.0) would produce, so there is no systematic overload of particular doctors. "
         f"Appointments per nurse range {(K.Appointments / K.Nurses).min():.1f}–{(K.Appointments / K.Nurses).max():.1f} across facilities (every facility has {int(K.Nurses.min())}–{int(K.Nurses.max())} nurses).")

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
ks = K.sort_values("Appointments", ascending=False)
ax[0].bar(ks.index.str[-2:], ks.Appointments, color=C[0], width=.7); ax[0].axhline(ks.Appointments.mean(), color=INK2, ls="--", lw=1)
ax[0].set(title="Facility utilisation: appointments by facility", xlabel="Facility (HF0xx)", ylabel="Appointments in 2026")
ax[1].scatter(K.TotalStaff, K.UniquePatients, s=60, color=C[0], edgecolor=SURF, zorder=3)
for f_, r_ in K.iterrows(): ax[1].annotate(f_[-2:], (r_.TotalStaff, r_.UniquePatients), textcoords="offset points", xytext=(4, 3), fontsize=7.5, color=INK2)
rr = stats.spearmanr(K.TotalStaff, K.UniquePatients)
ax[1].set(title="Staff vs patients served (by facility)", xlabel="Total staff (all cadres)", ylabel="Unique patients seen")
kk = K.sort_values("StaffPer1000Pts", ascending=False)
ax[2].barh(kk.index.str[-2:][::-1], kk.StaffPer1000Pts[::-1], color=C[2], height=.7); ax[2].set(title="Staff per 1,000 patients seen", xlabel="Staff per 1,000 unique patients", ylabel="Facility (HF0xx)")
fig.tight_layout()
R["fac_appt_range"] = [int(K.Appointments.min()), int(K.Appointments.max())]; R["staff_per1000_range"] = [float(K.StaffPer1000Pts.min()), float(K.StaffPer1000Pts.max())]
R["staff_vs_patients_rho"], R["staff_vs_patients_p"] = float(rr[0]), float(rr[1])
busy, quiet = ks.index[0], ks.index[-1]
fac_chi = stats.chisquare(K.Appointments.values)
R["fac_uniform_p"] = float(fac_chi.pvalue); R["busiest_facility"], R["quietest_facility"] = busy, quiet
sig11 = "differ significantly" if fac_chi.pvalue < 0.05 else "are not significantly different from equal shares"
save_fig(fig, "f11_facility_utilisation",
         f"Facility volumes run from {int(K.Appointments.min())} ({quiet}) to {int(K.Appointments.max())} ({busy}) appointments — a {pct(K.Appointments.max() / K.Appointments.min() - 1, 0)} gap — which {sig11} (χ² {fmt_p(fac_chi.pvalue)}; borderline). "
         f"Staffing is nearly identical everywhere ({int(K.TotalStaff.min())}–{int(K.TotalStaff.max())} staff per facility), so the load per staff member differs only through patient volume: {K.StaffPer1000Pts.min():.0f} to {K.StaffPer1000Pts.max():.0f} staff per 1,000 patients seen, "
         f"and staff numbers do not track patient volume (Spearman ρ = {rr[0]:.2f}, {fmt_p(rr[1])}).")

# %%
occ_sys = adm_all.LengthOfStayDays.sum() / (len(D["Beds"]) * 365)
occ_fac = (E.groupby("FacilityID").LengthOfStayDays.sum() / (K.Beds * 365)).fillna(0) * 100
dh = pd.crosstab(E.DepartmentName, E.Month.dt.month)
dh = dh.loc[DEPTT.sort_values("Appointments", ascending=False).index]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1, 1.1, 1.3]})
ax[0].bar(occ_fac.index.str[-2:], occ_fac.values, color=C[1], width=.7)
ax[0].axhline(100 * occ_sys, color=INK2, ls="--", lw=1); ax[0].text(0, 100 * occ_sys + .03, f"system estimate {100 * occ_sys:.1f}% (all 5,000 admissions)", fontsize=8, color=INK2)
ax[0].set(title="Bed occupancy – lower-bound estimate", xlabel="Facility (HF0xx)", ylabel="Occupied bed-days ÷ bed capacity (%)")
eq = D["Medical_Equipment"].EquipmentName.str.replace(" Equipment", "").value_counts()
ax[1].barh(eq.index[::-1], eq.values[::-1], color=C[0], height=.65); ax[1].set(title="Equipment items by department type", xlabel="Items (all facilities)")
for i, v in enumerate(eq.values[::-1]): ax[1].text(v + .3, i, str(v), va="center", fontsize=8.5)
im = ax[2].imshow(dh.values, aspect="auto", cmap=SEQ); ax[2].set_yticks(range(len(dh))); ax[2].set_yticklabels(dh.index); ax[2].set_xticks(range(12)); ax[2].set_xticklabels(range(1, 13))
ax[2].set(title="Appointments by department type and month", xlabel="Month (2026)"); ax[2].grid(False); fig.colorbar(im, ax=ax[2], fraction=.04, pad=.02, label="Appointments")
fig.tight_layout()
R["bed_occupancy_system"] = float(occ_sys); R["beds_total"] = int(len(D["Beds"])); R["beds_available_status"] = int((D["Beds"].Status == "Available").sum())
R["equip_per_dept_type"] = 20
save_fig(fig, "f12_capacity",
         f"All 2,000 beds (100 per facility, 5 per room) carry the status 'Available', so occupancy cannot be read directly. Using admission length-of-stay, recorded admissions fill only about {100 * occ_sys:.1f}% of annual bed-days at system level — "
         "far below any realistic figure, which suggests the Admissions extract is a sample. Equipment is one generic line per department (20 per department type), so equipment shortages cannot be measured. "
         "The department-by-month heatmap shows no obvious seasonal surge in any department type.")

# %% [markdown]
# ### 3.4 Financial analysis

# %%
rs = D["Revenue_Summary"].copy(); rs["Month"] = pd.to_datetime(rs.Month + "-01"); rs["RevPerAppt"] = rs.Revenue / rs.Appointments
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].plot(rs.Month, rs.Revenue / 1e6, color=C[0], lw=2, marker="o", ms=4); ax[0].axhline(rs.Revenue.mean() / 1e6, color=INK2, ls="--", lw=1)
ax[0].set(title="Monthly revenue (Revenue_Summary)", xlabel="Month (2026)", ylabel="Revenue (million RWF)")
imax, imin = rs.Revenue.idxmax(), rs.Revenue.idxmin()
ax[0].annotate(f"peak {rs.Month[imax]:%b}\n{rs.Revenue[imax] / 1e6:.0f} M", (rs.Month[imax], rs.Revenue[imax] / 1e6), textcoords="offset points", xytext=(-35, -6), ha="right", fontsize=8.5)
ax[0].annotate(f"low {rs.Month[imin]:%b}\n{rs.Revenue[imin] / 1e6:.0f} M", (rs.Month[imin], rs.Revenue[imin] / 1e6), textcoords="offset points", xytext=(0, -26), ha="center", fontsize=8.5)
ax[1].plot(rs.Month, rs.Appointments, color=C[1], lw=2, marker="o", ms=4); ax[1].set(title="Monthly appointments (Revenue_Summary)", xlabel="Month (2026)", ylabel="Appointments")
ax[2].plot(rs.Month, rs.RevPerAppt / 1e3, color=C[2], lw=2, marker="o", ms=4); ax[2].set(title="Revenue per appointment", xlabel="Month (2026)", ylabel="Thousand RWF per appointment")
for a_ in ax: a_.tick_params(axis="x", rotation=40)
fig.tight_layout()
rev_cv = rs.Revenue.std() / rs.Revenue.mean()
rev_app_r = stats.pearsonr(rs.Revenue, rs.Appointments); R["rev_appt_corr"] = [float(rev_app_r[0]), float(rev_app_r[1])]
R["rev_total"] = float(rs.Revenue.sum()); R["rev_mean"] = float(rs.Revenue.mean()); R["rev_cv"] = float(rev_cv); R["rev_peak"] = [f"{rs.Month[imax]:%b}", float(rs.Revenue[imax])]; R["rev_low"] = [f"{rs.Month[imin]:%b}", float(rs.Revenue[imin])]
R["appts_summary_total"] = int(rs.Appointments.sum())
save_fig(fig, "f13_revenue_trend",
         f"Revenue totals {rs.Revenue.sum() / 1e9:.2f} bn RWF for 2026 (mean {rs.Revenue.mean() / 1e6:.0f} M/month) but is volatile — coefficient of variation {pct(rev_cv, 0)}, from {rs.Revenue[imin] / 1e6:.0f} M in {rs.Month[imin]:%B} to {rs.Revenue[imax] / 1e6:.0f} M in {rs.Month[imax]:%B}. "
         f"Revenue per appointment swings between {rs.RevPerAppt.min() / 1e3:.0f}k and {rs.RevPerAppt.max() / 1e3:.0f}k RWF, and monthly revenue is only weakly related to monthly appointments (r = {rev_app_r[0]:.2f}, {fmt_p(rev_app_r[1])}), so revenue is driven by billing size at least as much as by visit counts.")

# %%
paid_tot, bill_tot, out_tot = F.AmountPaid.sum(), F.Amount.sum(), F.Outstanding.sum()
fig, ax = plt.subplots(2, 3, figsize=(16, 8.6))
ax[0, 0].hist(F.Amount / 1e3, bins=40, color=C[0], edgecolor=SURF, density=True); ax[0, 0].axvline(F.Amount.median() / 1e3, color=INK2, ls="--", lw=1)
ax[0, 0].set(title="Billing amount distribution (60,000 bills)", xlabel="Bill amount (thousand RWF)", ylabel="Density")
ax[0, 1].bar(["Billed", "Paid", "Outstanding"], np.array([bill_tot, paid_tot, out_tot]) / 1e9, color=[C[0], C[2], C[7]], width=.55)
for i, v in enumerate([bill_tot, paid_tot, out_tot]): ax[0, 1].text(i, v / 1e9 + .1, f"{v / 1e9:.2f} bn", ha="center", fontsize=9)
ax[0, 1].set(title="Billed vs paid vs outstanding (RWF bn)", ylabel="RWF billion", ylim=(0, bill_tot / 1e9 * 1.12))
pm = F.groupby("Method").agg(n=("PaymentID", "size"), paid=("AmountPaid", "sum"), out=("Outstanding", "sum")).sort_values("n", ascending=False)
ax[0, 2].bar(pm.index, 100 * pm.n / pm.n.sum(), color=C[0], width=.6)
for i, v in enumerate(pm.n): ax[0, 2].text(i, 100 * v / pm.n.sum() + .4, pct(v / pm.n.sum()), ha="center", fontsize=9)
ax[0, 2].set(title="Payment-method distribution", ylabel="% of payments", ylim=(0, 32))
cp = clm.groupby("Provider").agg(n=("ClaimID", "size"), amt=("ClaimAmount", "sum"), appr=("Status", lambda s: (s == "Approved").mean())).sort_values("amt", ascending=False)
ax[1, 0].bar(cp.index, cp.amt / 1e9, color=C[0], width=.6)
for i, v in enumerate(cp.amt): ax[1, 0].text(i, v / 1e9 + .02, f"{v / 1e9:.2f} bn", ha="center", fontsize=9)
ax[1, 0].set(title="Claim value by insurance provider", ylabel="RWF billion", ylim=(0, cp.amt.max() / 1e9 * 1.15))
ax[1, 1].bar(cp.index, 100 * cp.appr, color=C[2], width=.6); ax[1, 1].axhline(100 * (clm.Status == "Approved").mean(), color=INK2, ls="--", lw=1)
for i, v in enumerate(cp.appr): ax[1, 1].text(i, 100 * v + 1, f"{100 * v:.1f}%", ha="center", fontsize=9)
ax[1, 1].set(title="Claim approval rate by provider", ylabel="% approved", ylim=(0, 70))
sp_ = F.groupby("Status").agg(n=("BillingID", "size"), out=("Outstanding", "sum"), paid_ratio=("PaidRatio", "mean"))
ax[1, 2].bar(sp_.index, sp_.out / 1e6, color=C[1], width=.55)
for i, (v, r_) in enumerate(zip(sp_.out, sp_.paid_ratio)): ax[1, 2].text(i, v / 1e6 + 8, f"{v / 1e6:.0f} M\n(avg {100 * r_:.0f}% paid)", ha="center", fontsize=8.5)
ax[1, 2].set(title="Outstanding balance by recorded bill status", ylabel="Outstanding (million RWF)", ylim=(0, sp_.out.max() / 1e6 * 1.25))
fig.tight_layout()
R["bill_total"], R["paid_total"], R["outstanding_total"] = float(bill_tot), float(paid_tot), float(out_tot)
R["outstanding_pct"] = float(out_tot / bill_tot); R["mean_bill"], R["median_bill"] = float(F.Amount.mean()), float(F.Amount.median())
R["claim_approval"] = float((clm.Status == "Approved").mean()); R["claim_value"] = float(clm.ClaimAmount.sum())
R["provider_claim_value"] = cp.amt.to_dict(); R["provider_approval"] = cp.appr.to_dict(); R["pay_method_share"] = (pm.n / pm.n.sum()).to_dict()
chi_prov_appr = stats.chi2_contingency(pd.crosstab(clm.Provider, clm.Status))
R["provider_approval_p"] = float(chi_prov_appr[1])
save_fig(fig, "f14_billing_claims_payments",
         f"Bills are uniformly spread between 10k and 300k RWF (mean {F.Amount.mean() / 1e3:.0f}k). Of {bill_tot / 1e9:.2f} bn billed, {paid_tot / 1e9:.2f} bn has been paid, leaving {out_tot / 1e9:.2f} bn ({pct(out_tot / bill_tot)}) outstanding. "
         f"The recorded Status field is not a reliable guide: bills marked 'Paid', 'Partially Paid' and 'Pending' have the same average payment ratio (≈{100 * sp_.paid_ratio.mean():.0f}% in each group). "
         f"{cp.index[0]} generates the highest claim value ({cp.amt.iloc[0] / 1e9:.2f} bn). Only {pct((clm.Status == 'Approved').mean(), 0)} of claims are approved, and approval does not differ significantly across providers (χ² {fmt_p(chi_prov_appr[1])}).")

# %%
dep_rev = E.groupby("DepartmentName").agg(Revenue=("BillAmount", "sum"), Bills=("BillAmount", "size"), Mean=("BillAmount", "mean"), Sd=("BillAmount", "std")).sort_values("Revenue", ascending=False)
dep_rev["CI"] = 1.96 * dep_rev.Sd / np.sqrt(dep_rev.Bills)
fac_rev = E.groupby("FacilityID").agg(Revenue=("BillAmount", "sum"), Mean=("BillAmount", "mean")).sort_values("Revenue", ascending=False)
svc = E.assign(ServiceType="Consultation/Procedure")  # services are not linked to appointments -> documented
srv = E.groupby("Diagnosis").BillAmount.agg(["sum", "mean", "count"]).sort_values("sum", ascending=False)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
ax[0].barh(dep_rev.index[::-1], dep_rev.Revenue[::-1] / 1e6, color=C[0], height=.65)
for i, v in enumerate(dep_rev.Revenue[::-1]): ax[0].text(v / 1e6 + 1, i, f"{v / 1e6:.0f} M", va="center", fontsize=8.5)
ax[0].set(title="Billed revenue by department type (5,000-appointment cohort)", xlabel="Million RWF", xlim=(0, dep_rev.Revenue.max() / 1e6 * 1.15))
ax[1].bar(fac_rev.index.str[-2:], fac_rev.Revenue / 1e6, color=C[0], width=.7); ax[1].axhline(fac_rev.Revenue.mean() / 1e6, color=INK2, ls="--", lw=1)
ax[1].set(title="Billed revenue by facility (cohort)", xlabel="Facility (HF0xx)", ylabel="Million RWF")
ax[2].barh(srv.index[::-1], srv["sum"][::-1] / 1e6, color=C[2], height=.6)
for i, v in enumerate(srv["sum"][::-1]): ax[2].text(v / 1e6 + 1, i, f"{v / 1e6:.0f} M", va="center", fontsize=8.5)
ax[2].set(title="Billed revenue by diagnosis (service-line proxy)", xlabel="Million RWF", xlim=(0, srv["sum"].max() / 1e6 * 1.15))
fig.tight_layout()
kw_dep = stats.kruskal(*[g_.BillAmount.values for _, g_ in E.groupby("DepartmentName")])
kw_fac = stats.kruskal(*[g_.BillAmount.values for _, g_ in E.groupby("FacilityID")])
kw_dx = stats.kruskal(*[g_.BillAmount.values for _, g_ in E.groupby("Diagnosis")])
R["dep_rev_top"] = [dep_rev.index[0], float(dep_rev.Revenue.iloc[0])]; R["dep_rev_kw_p"] = float(kw_dep.pvalue); R["fac_rev_kw_p"] = float(kw_fac.pvalue); R["dx_rev_kw_p"] = float(kw_dx.pvalue)
R["fac_rev_top"] = [fac_rev.index[0], float(fac_rev.Revenue.iloc[0])]; R["cohort_billed"] = float(E.BillAmount.sum())
save_fig(fig, "f15_revenue_by_unit",
         f"Within the 5,000-appointment cohort ({E.BillAmount.sum() / 1e6:.0f} M RWF billed), {dep_rev.index[0]} bills the most ({dep_rev.Revenue.iloc[0] / 1e6:.0f} M) and {fac_rev.index[0]} is the top facility ({fac_rev.Revenue.iloc[0] / 1e6:.0f} M). "
         f"Differences are driven by appointment counts, not price: average bill per appointment does not differ significantly across departments (Kruskal-Wallis {fmt_p(kw_dep.pvalue)}), facilities ({fmt_p(kw_fac.pvalue)}) or diagnoses ({fmt_p(kw_dx.pvalue)}). "
         "Services are not linked to appointments in the data, so diagnosis is used as the service-line proxy; true 'revenue by service' needs a ServiceID on the bill.")

# %% [markdown]
# ### 3.5 Patient experience

# %%
fb_u = fb.copy()
rate_d = fb_u.Rating.value_counts().sort_index()
rc = pd.crosstab(fb_u.Rating, fb_u.Comment, normalize="index").loc[:, ["Poor", "Average", "Good", "Excellent"]] * 100
chi_rc = stats.chi2_contingency(pd.crosstab(fb_u.Rating, fb_u.Comment))
Ef = E[E.Rating.notna()].copy()
mf = Ef.groupby("Month").agg(poor=("Rating", lambda s: (s <= 2).mean()), exc=("Rating", lambda s: (s >= 4).mean()), mean=("Rating", "mean"), n=("Rating", "size"))
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1, 1.1, 1.1]})
ax[0].bar(rate_d.index.astype(str), rate_d.values, color=C[0], width=.65)
for i, v in enumerate(rate_d.values): ax[0].text(i, v + 60, pct(v / len(fb_u)), ha="center", fontsize=9)
ax[0].set(title="Satisfaction rating distribution (21,064)", xlabel="Rating (1 = lowest, 5 = highest)", ylabel="Feedback records", ylim=(0, rate_d.max() * 1.12))
im = ax[1].imshow(rc.values, cmap=SEQ, aspect="auto", vmin=15, vmax=35)
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(rc.columns); ax[1].set_yticks(range(5)); ax[1].set_yticklabels(rc.index); ax[1].grid(False)
for i in range(5):
    for j in range(4): ax[1].text(j, i, f"{rc.values[i, j]:.0f}%", ha="center", va="center", fontsize=9, color="white" if rc.values[i, j] > 27 else INK)
ax[1].set(title="Written comment given each rating (row %)", xlabel="Comment", ylabel="Rating")
ax[2].plot(mf.index, 100 * mf.poor, color=C[7], lw=2, marker="o", ms=4, label="Poor (rating 1–2)"); ax[2].plot(mf.index, 100 * mf.exc, color=C[2], lw=2, marker="s", ms=4, label="Excellent (rating 4–5)")
ax[2].set(title="Poor vs excellent rating trend (cohort, n=1,756)", xlabel="Month (2026)", ylabel="% of rated visits", ylim=(20, 70)); ax[2].legend(loc="upper center", ncol=2, bbox_to_anchor=(.5, -.2))
for t_ in ax[2].get_xticklabels(): t_.set_rotation(40)
fig.tight_layout()
tr_p = stats.spearmanr(np.arange(len(mf)), mf.poor); tr_e = stats.spearmanr(np.arange(len(mf)), mf.exc)
R["mean_rating"] = float(fb_u.Rating.mean()); R["rating_dist"] = (rate_d / len(fb_u)).to_dict(); R["rating_comment_p"] = float(chi_rc[1])
R["feedback_coverage"] = float(len(fb_u) / len(bill)); R["poor_trend"] = [float(tr_p[0]), float(tr_p[1])]; R["exc_trend"] = [float(tr_e[0]), float(tr_e[1])]
save_fig(fig, "f16_satisfaction",
         f"The average rating is {fb_u.Rating.mean():.2f} out of 5 and all five ratings are about equally common (≈20% each); {pct(len(fb_u) / len(bill), 0)} of encounters have feedback. "
         f"The written comment tells us nothing about the score — a rating of 1 is as likely to carry 'Excellent' as 'Poor' (χ² {fmt_p(chi_rc[1])}) — so feedback capture should be reviewed. "
         f"Monthly shares of poor and excellent ratings fluctuate without trend (Spearman p = {tr_p[1]:.2f} and {tr_e[1]:.2f}).")

# %%
fs = Ef.groupby("FacilityID").Rating.agg(["mean", "std", "count"]); fs["ci"] = 1.96 * fs["std"] / np.sqrt(fs["count"]); fs = fs.sort_values("mean", ascending=False)
ds = Ef.groupby("DepartmentName").Rating.agg(["mean", "std", "count"]); ds["ci"] = 1.96 * ds["std"] / np.sqrt(ds["count"]); ds = ds.sort_values("mean", ascending=False)
fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
ax[0].errorbar(fs["mean"][::-1], range(len(fs)), xerr=fs.ci[::-1], fmt="o", color=C[0], ecolor=MUTED, elinewidth=1.2, capsize=2.5, ms=6)
ax[0].set_yticks(range(len(fs))); ax[0].set_yticklabels(fs.index[::-1]); ax[0].axvline(Ef.Rating.mean(), color=INK2, ls="--", lw=1)
ax[0].set(title="Mean satisfaction by facility (95% CI)", xlabel="Mean rating (1–5)", xlim=(2.2, 3.9))
ax[1].errorbar(ds["mean"][::-1], range(len(ds)), xerr=ds.ci[::-1], fmt="o", color=C[1], ecolor=MUTED, elinewidth=1.2, capsize=2.5, ms=6)
ax[1].set_yticks(range(len(ds))); ax[1].set_yticklabels(ds.index[::-1]); ax[1].axvline(Ef.Rating.mean(), color=INK2, ls="--", lw=1)
ax[1].set(title="Mean satisfaction by department type (95% CI)", xlabel="Mean rating (1–5)", xlim=(2.2, 3.9))
fig.tight_layout()
kw_fs = stats.kruskal(*[g_.Rating.values for _, g_ in Ef.groupby("FacilityID")]); kw_ds = stats.kruskal(*[g_.Rating.values for _, g_ in Ef.groupby("DepartmentName")])
R["sat_fac_p"], R["sat_dept_p"] = float(kw_fs.pvalue), float(kw_ds.pvalue)
R["sat_best_fac"] = [fs.index[0], float(fs["mean"].iloc[0])]; R["sat_worst_fac"] = [fs.index[-1], float(fs["mean"].iloc[-1])]
R["sat_best_dept"] = [ds.index[0], float(ds["mean"].iloc[0])]; R["sat_worst_dept"] = [ds.index[-1], float(ds["mean"].iloc[-1])]
all_overlap = bool(((fs["mean"] - Ef.Rating.mean()).abs() <= fs.ci).all())
save_fig(fig, "f17_satisfaction_comparison",
         f"Facility means run from {fs['mean'].iloc[-1]:.2f} ({fs.index[-1]}) to {fs['mean'].iloc[0]:.2f} ({fs.index[0]}) and department-type means from {ds['mean'].iloc[-1]:.2f} to {ds['mean'].iloc[0]:.2f}, but with only ~{int(fs['count'].median())} rated visits per facility {'every confidence interval includes' if all_overlap else 'most confidence intervals include'} the overall mean "
         f"({Ef.Rating.mean():.2f}). Kruskal-Wallis finds no significant difference across facilities ({fmt_p(kw_fs.pvalue)}) or departments ({fmt_p(kw_ds.pvalue)}). Rankings of 'best' and 'worst' sites are therefore not statistically defensible with the current data.")

save(globals(), __file__)
print("Done:", os.path.basename(__file__))
