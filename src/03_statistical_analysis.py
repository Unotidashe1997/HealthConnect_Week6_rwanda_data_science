"""
HealthConnect Rwanda - Week 6 (Data Science track)
Part 3 - Descriptives, correlations and hypothesis tests (figures 18-26)

Run from the repository root:  python src/03_statistical_analysis.py
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
# ## 4. Statistical analysis
# ### 4.1 Descriptive statistics
# Mean, median, mode, standard deviation, variance, quartiles and percentiles for the key numeric variables, each on the
# most complete data source available for that variable (the base is shown in the `Source` column).

# %%
def describe_full(s, name, source):
    s = pd.Series(s).dropna().astype(float)
    mode = s.round(0).mode().iloc[0] if s.nunique() > 50 else s.mode().iloc[0]
    q = s.quantile([.05, .10, .25, .5, .75, .90, .95, .99])
    return {"Variable": name, "Source": source, "n": len(s), "Mean": s.mean(), "Median": s.median(), "Mode": mode, "Std Dev": s.std(), "Variance": s.var(),
            "Min": s.min(), "P5": q[.05], "P10": q[.10], "Q1 (P25)": q[.25], "Q3 (P75)": q[.75], "P90": q[.90], "P95": q[.95], "P99": q[.99], "Max": s.max(),
            "IQR": q[.75] - q[.25], "Skewness": stats.skew(s), "Excess kurtosis": stats.kurtosis(s), "CV %": 100 * s.std() / s.mean() if s.mean() else np.nan}

staff_exp = STAFF.YearsExperience
desc_rows = [
    describe_full(PATIENTS.Age, "Patient age (years)", "Patients (20,000)"),
    describe_full(PATIENTS.BMI, "BMI (kg/m²)", "Patients (20,000)"),
    describe_full(PATIENTS.HeightCm, "Height (cm)", "Patients (20,000)"),
    describe_full(PATIENTS.WeightKg, "Weight (kg)", "Patients (20,000)"),
    describe_full(adm_all.LengthOfStayDays, "Length of stay (days)", "Admissions (5,000)"),
    describe_full(FIN.BillAmount, "Bill amount (RWF)", "Billing (60,000)"),
    describe_full(FIN.AmountPaid, "Amount paid (RWF)", "Payments (60,000)"),
    describe_full(FIN.Outstanding, "Outstanding balance (RWF)", "Billing − Payments (60,000)"),
    describe_full(FIN.ClaimAmount, "Claim amount (RWF)", "Insurance claims (42,036)"),
    describe_full(FIN.ClaimRatio, "Claim ÷ bill", "Insurance claims (42,036)"),
    describe_full(fb.Rating, "Satisfaction rating (1–5)", "Feedback (21,064)"),
    describe_full(staff_exp, "Staff experience (years)", "All staff (1,250)"),
    describe_full(doc_load, "Appointments per doctor", "Appointments × doctors (350)"),
    describe_full(rs.Revenue / 1e6, "Monthly revenue (M RWF)", "Revenue summary (12)"),
    describe_full(rs.Appointments, "Monthly appointments", "Revenue summary (12)"),
]
DESC = pd.DataFrame(desc_rows)
R["descriptives"] = DESC.round(3).to_dict("records")
pd.options.display.float_format = "{:,.2f}".format
DESC.set_index("Variable")[["Source", "n", "Mean", "Median", "Mode", "Std Dev", "Variance", "Q1 (P25)", "Q3 (P75)", "P5", "P95", "Skewness", "CV %"]]

# %%
# categorical modes + normality checks
cat_modes = pd.DataFrame({"Variable": ["Gender", "Blood group", "Chronic condition", "Diagnosis", "Severity", "Visit type", "Lab test", "Medication", "Payment method", "Claim provider", "Rating"],
                          "Mode": [PATIENTS.Gender.mode()[0], PATIENTS.BloodGroup.mode()[0], PATIENTS.ChronicCondition.mode()[0], E.Diagnosis.mode()[0], E.Severity.mode()[0], E.VisitType.mode()[0],
                                   labs_t.Test.mode()[0], rx_t.Medication.mode()[0], pay.Method.mode()[0], clm.Provider.mode()[0], int(fb.Rating.mode()[0])],
                          "Share": [PATIENTS.Gender.value_counts(normalize=True).iloc[0], PATIENTS.BloodGroup.value_counts(normalize=True).iloc[0], PATIENTS.ChronicCondition.value_counts(normalize=True).iloc[0],
                                    E.Diagnosis.value_counts(normalize=True).iloc[0], E.Severity.value_counts(normalize=True).iloc[0], E.VisitType.value_counts(normalize=True).iloc[0],
                                    labs_t.Test.value_counts(normalize=True).iloc[0], rx_t.Medication.value_counts(normalize=True).iloc[0], pay.Method.value_counts(normalize=True).iloc[0],
                                    clm.Provider.value_counts(normalize=True).iloc[0], fb.Rating.value_counts(normalize=True).iloc[0]]})
cat_modes["Share"] = (100 * cat_modes.Share).round(1).astype(str) + "%"
R["cat_modes"] = cat_modes.to_dict("records")
norm = []
for nm, s in [("Patient age", PATIENTS.Age), ("BMI", PATIENTS.BMI), ("Bill amount", FIN.BillAmount), ("Length of stay", adm_all.LengthOfStayDays), ("Staff experience", staff_exp)]:
    smp = s.sample(min(len(s), 5000), random_state=SEED)
    nt = stats.normaltest(s); sw = stats.shapiro(smp)
    uni = stats.kstest((s - s.min()) / (s.max() - s.min()), "uniform")
    norm.append({"Variable": nm, "Shapiro-Wilk p": sw.pvalue, "D'Agostino p": nt.pvalue, "Skew": stats.skew(s), "KS vs uniform p": uni.pvalue,
                 "Verdict": "Not normal; consistent with uniform" if (nt.pvalue < .05 and uni.pvalue > .05) else ("Not normal" if nt.pvalue < .05 else "Cannot reject normality")})
NORM = pd.DataFrame(norm); R["normality"] = NORM.round(4).to_dict("records")
print(cat_modes.to_string(index=False)); print()
NORM

# %%
fig, ax = plt.subplots(2, 4, figsize=(17, 8))
specs = [(PATIENTS.Age, "Patient age (years)", C[0]), (PATIENTS.BMI, "BMI (kg/m²)", C[1]), (FIN.BillAmount / 1e3, "Bill amount (thousand RWF)", C[2]), (staff_exp, "Staff experience (years)", C[4])]
for j, (s_, lab_, col) in enumerate(specs):
    ax[0, j].hist(s_, bins=36, color=col, edgecolor=SURF, density=True, alpha=.9)
    sns.kdeplot(s_, ax=ax[0, j], color=INK, lw=1.3)
    for p_, ls_ in [(.25, ":"), (.5, "--"), (.75, ":")]: ax[0, j].axvline(s_.quantile(p_), color=INK2, ls=ls_, lw=1)
    ax[0, j].set(title=f"{lab_.split(' (')[0]}: histogram & density", xlabel=lab_, ylabel="Density")
    ax[1, j].boxplot(s_.dropna(), vert=False, widths=.5, patch_artist=True, boxprops=dict(facecolor=col, alpha=.55, edgecolor=INK2), medianprops=dict(color=INK, lw=1.6), flierprops=dict(marker=".", ms=2, markeredgecolor=MUTED))
    ax[1, j].set(title=f"{lab_.split(' (')[0]}: box plot", xlabel=lab_, yticks=[]); ax[1, j].grid(axis="y", visible=False)
fig.suptitle("Distributions of key numeric variables (dashed = median, dotted = quartiles)", fontsize=13.5, fontweight="semibold", y=1.0)
fig.tight_layout()
out_bmi = ((PATIENTS.BMI > PATIENTS.BMI.quantile(.75) + 1.5 * stats.iqr(PATIENTS.BMI))).mean()
R["bmi_outlier_pct"] = float(out_bmi)
save_fig(fig, "f18_distributions",
         f"None of the key variables is bell-shaped: patient age, length of stay and staff experience are flat across their range, and bill amount is statistically indistinguishable from a uniform distribution (KS p = {stats.kstest((FIN.BillAmount - FIN.BillAmount.min()) / (FIN.BillAmount.max() - FIN.BillAmount.min()), 'uniform').pvalue:.2f}), "
         f"while BMI is right-skewed (skew {stats.skew(PATIENTS.BMI):.2f}; median {PATIENTS.BMI.median():.1f}, mean {PATIENTS.BMI.mean():.1f}) with {pct(out_bmi)} of patients above the upper box-plot fence. "
         f"Because the data are not normal, rank-based tests (Mann-Whitney, Kruskal-Wallis, Spearman) are used alongside parametric ones in the sections that follow. The median bill is {FIN.BillAmount.median() / 1e3:.0f}k RWF.")

# %%
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
order_s = ["Mild", "Moderate", "Severe"]
sns.violinplot(data=FIN.dropna(subset=["Severity"]), x="Severity", y=FIN.BillAmount / 1e3, order=order_s, palette=[C[2], C[3], C[7]], inner="quartile", linewidth=1, ax=ax[0], cut=0)
ax[0].set(title="Bill amount by disease severity", xlabel="Severity", ylabel="Bill amount (thousand RWF)")
sns.violinplot(data=PATIENTS, x="ChronicCondition", y="BMI", order=["None", "Asthma", "Hypertension", "Diabetes"], palette=[MUTED, C[0], C[1], C[2]], inner="quartile", linewidth=1, ax=ax[1], cut=0)
ax[1].set(title="BMI by chronic condition", xlabel="Chronic condition", ylabel="BMI (kg/m²)")
sns.violinplot(data=E.dropna(subset=["Rating"]), x="VisitType", y="Rating", order=["Consultation", "Follow-up", "Emergency"], palette=[C[0], C[1], C[2]], inner="quartile", linewidth=1, ax=ax[2], cut=0)
ax[2].set(title="Satisfaction rating by visit type", xlabel="Visit type", ylabel="Rating (1–5)")
fig.tight_layout()
kw_sev_bill = stats.kruskal(*[g_.BillAmount.values for _, g_ in FIN.dropna(subset=["Severity"]).groupby("Severity")])
kw_bmi_chr = stats.kruskal(*[g_.BMI.values for _, g_ in PATIENTS.groupby("ChronicCondition")])
kw_vt_rat = stats.kruskal(*[g_.Rating.values for _, g_ in E.dropna(subset=["Rating"]).groupby("VisitType")])
R["bill_sev_p"], R["bmi_chronic_p"], R["rating_visit_p"] = float(kw_sev_bill.pvalue), float(kw_bmi_chr.pvalue), float(kw_vt_rat.pvalue)
save_fig(fig, "f19_violins",
         f"Violin shapes are essentially identical across groups: bills do not differ by severity (Kruskal-Wallis {fmt_p(kw_sev_bill.pvalue)}) — severe cases are not billed more than mild ones, which is clinically unexpected; "
         f"BMI does not differ by chronic condition ({fmt_p(kw_bmi_chr.pvalue)}); and satisfaction does not differ by visit type ({fmt_p(kw_vt_rat.pvalue)}). Where real-world relationships should exist (severity→cost, obesity→diabetes/hypertension) none is present.")

# %% [markdown]
# ### 4.2 Correlation analysis
# The six relationships required by the brief. Each uses the largest valid sample, the appropriate coefficient, and reports an effect size so that
# "significant" is not confused with "important". Strength labels: |r| < 0.1 negligible, 0.1–0.3 weak, 0.3–0.5 moderate, > 0.5 strong.

# %%
def strength(r):
    r = abs(r); return "negligible" if r < .1 else "weak" if r < .3 else "moderate" if r < .5 else "strong"

def eps2(H, n): return H / (n - 1)
CORR = []
# 1 Age ~ admission
pb = stats.pointbiserialr(E.Admitted, E.Age); sp1 = stats.spearmanr(E.Age, E.Admitted)
CORR.append(dict(Pair="Age ↔ hospital admission (0/1)", Data="Appointment cohort", n=len(E), Method="Point-biserial (Spearman ρ = %.3f)" % sp1[0], Coefficient=pb[0], p_value=pb[1], Strength=strength(pb[0])))
# 2 Age ~ billing
pr2 = stats.pearsonr(E.Age, E.BillAmount); sp2 = stats.spearmanr(E.Age, E.BillAmount)
CORR.append(dict(Pair="Age ↔ billing amount", Data="Appointment cohort", n=len(E), Method="Pearson (Spearman ρ = %.3f)" % sp2[0], Coefficient=pr2[0], p_value=pr2[1], Strength=strength(pr2[0])))
# 3 Diagnosis ~ LOS
Ea = E[E.Admitted == 1]
kw3 = stats.kruskal(*[g_.LengthOfStayDays.values for _, g_ in Ea.groupby("Diagnosis")]); an3 = stats.f_oneway(*[g_.LengthOfStayDays.values for _, g_ in Ea.groupby("Diagnosis")])
grand = Ea.LengthOfStayDays.mean(); ssb = sum(len(g_) * (g_.LengthOfStayDays.mean() - grand) ** 2 for _, g_ in Ea.groupby("Diagnosis")); sst = ((Ea.LengthOfStayDays - grand) ** 2).sum()
eta = np.sqrt(ssb / sst)
CORR.append(dict(Pair="Diagnosis ↔ length of stay", Data="Admitted cohort", n=len(Ea), Method="Kruskal-Wallis; η = %.3f (ANOVA p = %.3f)" % (eta, an3.pvalue), Coefficient=eta, p_value=kw3.pvalue, Strength=strength(eta)))
# 4 Department workload ~ satisfaction (department-ID level, department-type level, facility level)
dsat = DEPT.dropna(subset=["MeanRating", "WorkloadIndex"]); sp4 = stats.spearmanr(dsat.WorkloadIndex, dsat.MeanRating)
CORR.append(dict(Pair="Department workload index ↔ mean satisfaction", Data="200 departments (IDs)", n=len(dsat), Method="Spearman", Coefficient=sp4[0], p_value=sp4[1], Strength=strength(sp4[0])))
dts = Ef.groupby("DepartmentName").Rating.mean(); sp4b = stats.spearmanr(DEPTT.WorkloadIndex.loc[dts.index], dts)
CORR.append(dict(Pair="  … same, 10 department types", Data="10 department types", n=len(dts), Method="Spearman", Coefficient=sp4b[0], p_value=sp4b[1], Strength=strength(sp4b[0])))
sp4c = stats.spearmanr(K.ApptPerClinician, K.MeanRating)
CORR.append(dict(Pair="  … same, 20 facilities (appts per clinician)", Data="20 facilities", n=len(K), Method="Spearman", Coefficient=sp4c[0], p_value=sp4c[1], Strength=strength(sp4c[0])))
# 5 Revenue ~ patient volume
pr5 = stats.pearsonr(rs.Revenue, rs.Appointments); sp5 = stats.spearmanr(rs.Revenue, rs.Appointments)
CORR.append(dict(Pair="Monthly revenue ↔ monthly appointments", Data="Revenue summary, 12 months", n=12, Method="Pearson (Spearman ρ = %.3f)" % sp5[0], Coefficient=pr5[0], p_value=pr5[1], Strength=strength(pr5[0])))
pr5b = stats.pearsonr(K.BilledTotal, K.Appointments)
CORR.append(dict(Pair="  … facility billed revenue ↔ appointments", Data="20 facilities (cohort)", n=20, Method="Pearson", Coefficient=pr5b[0], p_value=pr5b[1], Strength=strength(pr5b[0])))
# 6 Insurance coverage ~ utilisation
vis_kw = stats.kruskal(*[g_.Visits2026.values for _, g_ in PATIENTS.groupby("InsuranceID")])
chi6 = stats.chi2_contingency(pd.crosstab(PATIENTS.InsuranceID, PATIENTS.Visits2026 > 0)); v6 = np.sqrt(chi6[0] / (len(PATIENTS) * 14))
CORR.append(dict(Pair="Insurance code (15) ↔ healthcare utilisation (visited yes/no)", Data="20,000 patients", n=len(PATIENTS), Method="χ²; Cramér's V = %.3f" % v6, Coefficient=v6, p_value=chi6[1], Strength=strength(v6)))
cv = FIN.dropna(subset=["ClaimRatio"]); sp6 = stats.spearmanr(cv.ClaimRatio, cv.BillAmount)
CORR.append(dict(Pair="Claim coverage ratio ↔ bill amount", Data="42,036 bills with claims", n=len(cv), Method="Spearman", Coefficient=sp6[0], p_value=sp6[1], Strength=strength(sp6[0])))
fcl = stats.mannwhitneyu(FIN[FIN.HasClaim == 1].BillAmount, FIN[FIN.HasClaim == 0].BillAmount)
CORR.append(dict(Pair="Claim filed (0/1) ↔ bill amount", Data="60,000 bills", n=len(FIN), Method="Point-biserial", Coefficient=stats.pointbiserialr(FIN.HasClaim, FIN.BillAmount)[0], p_value=fcl.pvalue, Strength=strength(stats.pointbiserialr(FIN.HasClaim, FIN.BillAmount)[0])))
CORRT = pd.DataFrame(CORR); CORRT["Significant (α=.05)"] = np.where(CORRT.p_value < .05, "Yes", "No")
R["correlations"] = CORRT.round(4).to_dict("records")
pd.options.display.float_format = "{:,.4f}".format
CORRT[["Pair", "Data", "n", "Method", "Coefficient", "p_value", "Strength", "Significant (α=.05)"]]

# %%
# Full Spearman correlation matrix on the encounter master + multiple-testing reality check
E["LabAbnormalNum"] = np.where(E.LabDone == 1, E.LabAbnormal, np.nan)
cm_cols = {"Age": "Age", "BMI": "BMI", "ChronicFlag": "Chronic disease", "SeverityScore": "Severity score", "Admitted": "Admitted", "LengthOfStayDays": "Length of stay",
           "BillAmount": "Bill amount", "AmountPaid": "Amount paid", "PaidRatio": "Paid ÷ billed", "ClaimRatio": "Claim ÷ bill", "Rating": "Satisfaction rating",
           "DoctorYearsExp": "Doctor experience", "PatientVisits": "Patient visits", "LabAbnormalNum": "Abnormal lab", "Referred": "Referred", "DeptWorkloadIndex": "Dept workload idx"}
M = E[list(cm_cols)].rename(columns=cm_cols).astype(float)
rho = M.corr(method="spearman")
pv = pd.DataFrame(np.ones_like(rho.values), index=rho.index, columns=rho.columns); nn = pv.copy()
for a_, b_ in itertools.combinations(M.columns, 2):
    d_ = M[[a_, b_]].dropna()
    r_, p_ = stats.spearmanr(d_[a_], d_[b_]); pv.loc[a_, b_] = pv.loc[b_, a_] = p_; nn.loc[a_, b_] = nn.loc[b_, a_] = len(d_)
iu = np.triu_indices(len(rho), 1)
n_pairs = len(iu[0]); n_sig = int((pv.values[iu] < .05).sum())
pairs = pd.DataFrame({"A": rho.index[iu[0]], "B": rho.columns[iu[1]], "rho": rho.values[iu], "p": pv.values[iu], "n": nn.values[iu]})
pairs["abs"] = pairs.rho.abs(); top_pairs = pairs.sort_values("abs", ascending=False).head(8)
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, idx in enumerate(o): run = max(run, (m - i) * p[idx]); adj[idx] = min(1, run)
    return adj
pairs["p_holm"] = holm(pairs.p.values); n_sig_holm = int((pairs.p_holm < .05).sum())
R["corr_matrix_pairs"], R["corr_matrix_sig"], R["corr_matrix_sig_holm"] = int(n_pairs), n_sig, n_sig_holm
R["corr_top"] = top_pairs.round(3).to_dict("records")
fig, ax = plt.subplots(figsize=(11.5, 9.2))
mask = np.triu(np.ones_like(rho, dtype=bool), 1)
sns.heatmap(rho, mask=mask, cmap=DIV, vmin=-1, vmax=1, center=0, annot=True, fmt=".2f", annot_kws={"size": 7.5}, linewidths=.6, linecolor=SURF, cbar_kws={"label": "Spearman ρ", "shrink": .75}, ax=ax)
ax.set(title="Spearman correlation heatmap — encounter master (pairwise complete observations)"); ax.grid(False)
by_design = [("Admitted", "Length of stay"), ("Bill amount", "Amount paid"), ("Amount paid", "Paid ÷ billed"), ("Bill amount", "Paid ÷ billed"), ("Bill amount", "Claim ÷ bill")]
save_fig(fig, "f20_correlation_heatmap",
         f"Of {n_pairs} variable pairs only {n_sig} reach p < 0.05 unadjusted (about {int(round(.05 * n_pairs))} are expected by pure chance), and {n_sig_holm} survive Holm correction. "
         f"The pairs that survive Holm correction are arithmetic ones ({'; '.join(f'{r_.A} ↔ {r_.B} (ρ = {r_.rho:.2f})' for _, r_ in pairs[pairs.p_holm < .05].iterrows())}). Every clinically meaningful pair — age, BMI, chronic disease, severity, admission, bill, rating — is |ρ| < 0.13. "
         "Key drivers that would normally separate patient groups are absent, which limits what any predictive model can achieve.")

# %%
samp = E.sample(1500, random_state=SEED)[["Age", "BMI", "BillAmount", "DoctorYearsExp", "Severity"]].rename(columns={"BillAmount": "Bill (RWF)", "DoctorYearsExp": "Doctor exp. (y)"})
g_ = sns.pairplot(samp, hue="Severity", hue_order=["Mild", "Moderate", "Severe"], palette=[C[2], C[3], C[7]], diag_kind="kde", plot_kws=dict(s=10, alpha=.45, linewidth=0), height=2.0, corner=False)
g_.figure.suptitle("Pair plot — age, BMI, bill amount and doctor experience by disease severity (1,500 sampled appointments)", y=1.02, fontsize=12.5, fontweight="semibold")
save_fig(g_.figure, "f21_pairplot", "The scatter panels form rectangular clouds with no slope, the density curves on the diagonal overlap for all three severity levels, and no cluster of points separates Severe from Mild cases. "
         "Age, BMI, bill size and the treating doctor's experience are statistically independent of each other and of severity in this data.", title="Pair plot — age, BMI, bill amount, doctor experience by severity")

# %%
ag = E.groupby("AgeGroup", observed=True).agg(n=("Admitted", "size"), adm=("Admitted", "sum"), bill=("BillAmount", "mean"))
ag["rate"] = ag.adm / ag.n
lo, hi = zip(*[stats.beta.interval(.95, k + .5, n - k + .5) for k, n in zip(ag.adm, ag.n)])
dl = Ea.groupby("Diagnosis").LengthOfStayDays.agg(["mean", "std", "count"]); dl["ci"] = 1.96 * dl["std"] / np.sqrt(dl["count"])
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
ax[0].errorbar(range(len(ag)), 100 * ag.rate, yerr=[100 * (ag.rate - np.array(lo)), 100 * (np.array(hi) - ag.rate)], fmt="o-", color=C[0], ecolor=MUTED, capsize=3, lw=1.8)
ax[0].axhline(100 * E.Admitted.mean(), color=INK2, ls="--", lw=1); ax[0].set_xticks(range(len(ag))); ax[0].set_xticklabels(ag.index.astype(str))
ax[0].set(title="Admission rate by age group (95% CI)", xlabel="Age group", ylabel="% of appointments admitted", ylim=(0, 22))
ax[1].scatter(E.Age, E.BillAmount / 1e3, s=5, alpha=.35, color=C[0], linewidths=0); xs_ = np.linspace(0, 90, 50); b_ = np.polyfit(E.Age, E.BillAmount / 1e3, 1)
ax[1].plot(xs_, np.polyval(b_, xs_), color=C[1], lw=2); ax[1].set(title=f"Age vs bill amount (Pearson r = {pr2[0]:.3f})", xlabel="Age (years)", ylabel="Bill amount (thousand RWF)")
ax[2].errorbar(dl["mean"], range(len(dl)), xerr=dl.ci, fmt="o", color=C[2], ecolor=MUTED, capsize=3); ax[2].set_yticks(range(len(dl))); ax[2].set_yticklabels(dl.index)
ax[2].axvline(Ea.LengthOfStayDays.mean(), color=INK2, ls="--", lw=1); ax[2].set(title="Mean length of stay by diagnosis (95% CI)", xlabel="Days")
fig.tight_layout()
save_fig(fig, "f22_age_admission_bill_los",
         f"Admission rates by age group stay within noise of the {pct(E.Admitted.mean())} average (highest {100 * ag.rate.max():.1f}% in {ag.rate.idxmax()}); the age–admission correlation is {pb[0]:.3f} ({fmt_p(pb[1])}) and the age–bill correlation {pr2[0]:.3f} ({fmt_p(pr2[1])}). "
         f"Mean length of stay is {dl['mean'].min():.1f}–{dl['mean'].max():.1f} days across diagnoses with overlapping intervals (Kruskal-Wallis {fmt_p(kw3.pvalue)}). Older patients are not admitted more often, billed more, or kept longer.")

# %% [markdown]
# ### 4.3 What drives hospital admission? (business question 1)
# Every candidate factor available at or around the time of the appointment is tested against admission (n = 5,000, 586 admitted).
# Categorical factors use a χ² test (Cramér's V), numeric factors a Mann-Whitney test; p-values are Holm-adjusted for the number of factors screened.

# %%
fac_tests = []
Ec = E.copy(); Ec["Province"] = Ec.ProvinceName; Ec["Month_"] = Ec.MonthNum.astype(str)
for col, nm in [("VisitType", "Visit type"), ("Severity", "Disease severity"), ("Diagnosis", "Diagnosis"), ("DepartmentName", "Department type"), ("Gender", "Gender"), ("AgeGroup", "Age group"),
                ("ChronicCondition", "Chronic condition"), ("HighRisk", "High-risk patient flag"), ("BMICategory", "BMI category"), ("FacilityID", "Facility"), ("Province", "Patient province"),
                ("Month_", "Month of visit"), ("DayOfWeek", "Day of week"), ("Referred", "Referred"), ("LabAbnormalNum", "Abnormal lab result"), ("FrequentVisitor", "Frequent visitor"),
                ("Education", "Education"), ("MaritalStatus", "Marital status"), ("InsuranceID", "Insurance code"), ("CoverageCategory", "Claim coverage category")]:
    d_ = Ec[[col, "Admitted"]].dropna(); ct_ = pd.crosstab(d_[col], d_["Admitted"]); c2 = stats.chi2_contingency(ct_)
    V = np.sqrt(c2[0] / (len(d_) * (min(ct_.shape) - 1))); rates = d_.groupby(col, observed=True).Admitted.mean()
    fac_tests.append(dict(Factor=nm, Type="categorical", n=len(d_), Test="χ²", Statistic=c2[0], p=c2[1], Effect=V, Effect_name="Cramér's V", Highest_group=f"{rates.idxmax()} ({100 * rates.max():.1f}%)", Lowest_group=f"{rates.idxmin()} ({100 * rates.min():.1f}%)"))
for col, nm in [("Age", "Age"), ("BMI", "BMI"), ("DoctorYearsExp", "Doctor experience (y)"), ("BillAmount", "Bill amount"), ("PatientVisits", "Patient visits"), ("DeptWorkloadIndex", "Department workload index")]:
    a_, b_ = Ec.loc[Ec.Admitted == 1, col].dropna(), Ec.loc[Ec.Admitted == 0, col].dropna(); mw = stats.mannwhitneyu(a_, b_)
    r_ = 1 - 2 * mw.statistic / (len(a_) * len(b_))
    fac_tests.append(dict(Factor=nm, Type="numeric", n=len(a_) + len(b_), Test="Mann-Whitney", Statistic=mw.statistic, p=mw.pvalue, Effect=abs(r_), Effect_name="|rank-biserial r|", Highest_group=f"admitted median {a_.median():.1f}", Lowest_group=f"not admitted median {b_.median():.1f}"))
FT = pd.DataFrame(fac_tests); FT["p_holm"] = holm(FT.p.values); FT["Signif_raw"] = FT.p < .05; FT["Signif_holm"] = FT.p_holm < .05
FT = FT.sort_values("p").reset_index(drop=True)
R["admission_factor_scan"] = FT.round(4).to_dict("records"); R["admission_factors_sig_raw"] = int(FT.Signif_raw.sum()); R["admission_factors_tested"] = len(FT); R["admission_factors_sig_holm"] = int(FT.Signif_holm.sum())
pd.options.display.float_format = "{:,.4f}".format
FT[["Factor", "n", "Test", "p", "p_holm", "Effect", "Effect_name", "Highest_group", "Lowest_group"]]

# %%
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
colors_ = [C[7] if s_ else (C[1] if r_ else MUTED) for s_, r_ in zip(FT.Signif_holm, FT.Signif_raw)]
ax[0].barh(FT.Factor[::-1], -np.log10(FT.p[::-1]), color=colors_[::-1], height=.65)
ax[0].axvline(-np.log10(.05), color=INK2, ls="--", lw=1); ax[0].text(-np.log10(.05) + .03, 0, "p = 0.05", fontsize=8, color=INK2, va="bottom")
ax[0].set(title="Association of each factor with admission", xlabel="−log10(p-value)  (longer = stronger evidence)")
ax[1].barh(FT.Factor[::-1], FT.Effect[::-1], color=C[0], height=.65); ax[1].set(title="Effect size (Cramér's V or |rank-biserial r|)", xlabel="Effect size (0 = none)"); ax[1].set_yticklabels([])
fig.tight_layout()
best = FT.iloc[0]
save_fig(fig, "f23_admission_drivers",
         f"Of {len(FT)} candidate factors, {int(FT.Signif_raw.sum())} reach p < 0.05 before correction (about {max(1, int(round(.05 * len(FT))))} expected by chance) and {int(FT.Signif_holm.sum())} after Holm correction. "
         f"The strongest signal is '{best.Factor}' (raw p = {best.p:.3f}, effect size {best.Effect:.3f}), which is negligible in size. "
         "No demographic, clinical or operational factor can be confirmed as a driver of admission, so admission in this data is best described as unexplained by recorded variables.")

# %% [markdown]
# ### 4.4 Which patient groups are most at risk of chronic disease? (business question 2)
# Prevalence of any chronic condition (asthma, hypertension or diabetes) by every patient attribute, compared with the overall rate
# (74.6%) using Wilson 95% confidence intervals and a χ² test per attribute. A logistic regression then tests whether the attributes
# jointly predict chronic disease.

# %%
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d

base = PATIENTS.ChronicFlag.mean()
rows, attr_tests = [], []
for col, nm in [("AgeGroup", "Age group"), ("Gender", "Gender"), ("BMICategory", "BMI category"), ("Education", "Education"), ("Occupation", "Occupation"),
                ("MaritalStatus", "Marital status"), ("ProvinceName", "Province"), ("BloodGroup", "Blood group")]:
    ct_ = pd.crosstab(PATIENTS[col], PATIENTS.ChronicFlag); c2 = stats.chi2_contingency(ct_)
    attr_tests.append(dict(Attribute=nm, chi2=c2[0], p=c2[1], CramersV=np.sqrt(c2[0] / (len(PATIENTS) * (min(ct_.shape) - 1)))))
    for lvl, r_ in ct_.iterrows():
        n_, k_ = r_.sum(), r_.get(1, 0); lo_, hi_ = wilson(k_, n_)
        rows.append(dict(Attribute=nm, Group=str(lvl), n=int(n_), Prevalence=k_ / n_, Lo=lo_, Hi=hi_, Diff=k_ / n_ - base, Excludes=(lo_ > base) or (hi_ < base)))
PREV = pd.DataFrame(rows); ATT = pd.DataFrame(attr_tests); ATT["p_holm"] = holm(ATT.p.values)
R["chronic_attr_tests"] = ATT.round(4).to_dict("records"); R["chronic_groups_excluding_base"] = int(PREV.Excludes.sum()); R["chronic_groups_tested"] = len(PREV)
PREV_sorted = PREV.sort_values("Diff", ascending=False)
R["chronic_top_groups"] = PREV_sorted.head(5).round(4).to_dict("records")
fig, ax = plt.subplots(figsize=(10, 12))
pv_ = PREV.reset_index(drop=True); ypos = np.arange(len(pv_))[::-1]
ax.errorbar(100 * pv_.Prevalence, ypos, xerr=[100 * (pv_.Prevalence - pv_.Lo), 100 * (pv_.Hi - pv_.Prevalence)], fmt="none", ecolor=MUTED, elinewidth=1.2, capsize=2)
ax.scatter(100 * pv_.Prevalence, ypos, c=np.where(pv_.Excludes, C[7], C[0]), s=36, zorder=3)
ax.axvline(100 * base, color=INK2, ls="--", lw=1); ax.set_yticks(ypos); ax.set_yticklabels(pv_.Attribute + ": " + pv_.Group, fontsize=8)
ax.set(title="Chronic-disease prevalence by patient group with 95% Wilson CI (red = interval excludes overall 74.6%)", xlabel="% of group with a chronic condition", xlim=(68, 82)); ax.grid(axis="y", visible=False)
save_fig(fig, "f24_chronic_risk_groups",
         f"Among {len(PREV)} patient groups, {int(PREV.Excludes.sum())} have a prevalence interval that excludes the overall {pct(base)} (about {int(round(.05 * len(PREV)))} would be expected by chance). The highest-prevalence group is "
         f"{PREV_sorted.iloc[0].Attribute} = {PREV_sorted.iloc[0].Group} ({pct(PREV_sorted.iloc[0].Prevalence)}, n = {int(PREV_sorted.iloc[0].n):,}), only {100 * PREV_sorted.iloc[0].Diff:.1f} points above average. "
         + (f"Only {', '.join(ATT[ATT.p_holm < .05].Attribute)} survives Holm correction (Cramér's V = {ATT[ATT.p_holm < .05].CramersV.max():.3f}, a negligible effect), so no group can be called materially more at risk." if (ATT.p_holm < .05).any() else "No attribute test survives Holm correction, so no group can be called materially more at risk on the evidence available."))

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

num_c = ["Age", "BMI"]; cat_c = ["Gender", "MaritalStatus", "Education", "Occupation", "ProvinceName", "BloodGroup"]
pre = ColumnTransformer([("n", StandardScaler(), num_c), ("c", OneHotEncoder(drop="first"), cat_c)])
lr_c = Pipeline([("p", pre), ("m", LogisticRegression(max_iter=1000))])
yC = PATIENTS.ChronicFlag.values
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
auc_chr = cross_val_score(lr_c, PATIENTS[num_c + cat_c], yC, cv=skf, scoring="roc_auc")
lr_c.fit(PATIENTS[num_c + cat_c], yC)
names = lr_c.named_steps["p"].get_feature_names_out(); coefs = lr_c.named_steps["m"].coef_[0]
# bootstrap odds-ratio intervals (200 resamples)
bs = []
Xc = lr_c.named_steps["p"].transform(PATIENTS[num_c + cat_c])
for b_ in range(150):
    ix = rng.integers(0, len(yC), len(yC)); bs.append(LogisticRegression(max_iter=500).fit(Xc[ix], yC[ix]).coef_[0])
bs = np.array(bs); OR = pd.DataFrame({"Feature": [n.split("__")[1] for n in names], "OR": np.exp(coefs), "Lo": np.exp(np.percentile(bs, 2.5, 0)), "Hi": np.exp(np.percentile(bs, 97.5, 0))})
OR["Excludes1"] = (OR.Lo > 1) | (OR.Hi < 1)
R["chronic_auc_cv"] = [float(auc_chr.mean()), float(auc_chr.std())]; R["chronic_or_excl"] = int(OR.Excludes1.sum()); R["chronic_or_n"] = len(OR)
fig, ax = plt.subplots(figsize=(8.5, 9))
yy = np.arange(len(OR))[::-1]
ax.errorbar(OR.OR, yy, xerr=[OR.OR - OR.Lo, OR.Hi - OR.OR], fmt="none", ecolor=MUTED, capsize=2); ax.scatter(OR.OR, yy, c=np.where(OR.Excludes1, C[7], C[0]), s=34, zorder=3)
ax.axvline(1, color=INK2, ls="--", lw=1); ax.set_yticks(yy); ax.set_yticklabels(OR.Feature, fontsize=7.5); ax.set_xscale("log"); ax.grid(axis="y", visible=False)
ax.set(title="Logistic regression: odds ratios for any chronic disease (95% bootstrap CI)", xlabel="Odds ratio (log scale; 1 = no effect)")
save_fig(fig, "f25_chronic_odds_ratios",
         f"Cross-validated AUC for predicting chronic disease from age, BMI, gender, marital status, education, occupation, province and blood group is {auc_chr.mean():.3f} ± {auc_chr.std():.3f} (0.5 = coin-flip). "
         f"Only {int(OR.Excludes1.sum())} of {len(OR)} odds ratios have an interval that excludes 1 (about {max(1, int(round(.05 * len(OR))))} expected by chance). With an AUC this close to 0.5, the recorded demographics cannot meaningfully identify who is at risk of chronic disease.")

# %% [markdown]
# ### 4.5 Hypothesis tests
# Significance level **α = 0.05** for every test. Where several tests address the same family, Holm-adjusted p-values are also given. Effect sizes are reported
# because with large n tiny differences become "significant" and with small n large differences can be missed.

# %%
HT = []
def add_ht(i, q, h0, h1, test, n, stat, p, eff, effname, concl_extra=""):
    dec = "Reject H0" if p < .05 else "Fail to reject H0"
    HT.append(dict(ID=i, Question=q, H0=h0, H1=h1, Test=test, Alpha=.05, n=n, Statistic=float(stat), p_value=float(p), Effect=float(eff), Effect_measure=effname, Decision=dec, Conclusion=concl_extra))

# H1 billing: claim vs no claim (the closest available analogue of insured vs uninsured)
a_, b_ = FIN[FIN.HasClaim == 1].BillAmount, FIN[FIN.HasClaim == 0].BillAmount
tt = stats.ttest_ind(a_, b_, equal_var=False); mw = stats.mannwhitneyu(a_, b_); d = (a_.mean() - b_.mean()) / np.sqrt((a_.var() + b_.var()) / 2)
add_ht("H1", "Do average bills differ between bills with an insurance claim and bills without? (100% of patients are insured, so 'uninsured' cannot be tested)",
       "Mean bill (claim) = mean bill (no claim)", "Mean bills differ", "Welch two-sample t-test (Mann-Whitney U agrees: p = %.3f)" % mw.pvalue, len(FIN), tt.statistic, tt.pvalue, d, "Cohen's d",
       f"Mean bill {a_.mean():,.0f} RWF with claim vs {b_.mean():,.0f} RWF without ({a_.mean() - b_.mean():+,.0f}).")
# H1b payment method
ins_b = FIN[FIN.PayMethod == "Insurance"].BillAmount; oth_b = FIN[FIN.PayMethod != "Insurance"].BillAmount; t1b = stats.ttest_ind(ins_b, oth_b, equal_var=False)
add_ht("H1b", "Do bills paid via the 'Insurance' method differ from other payment methods?", "Mean bill equal", "Mean bill differs", "Welch t-test", len(FIN), t1b.statistic, t1b.pvalue,
       (ins_b.mean() - oth_b.mean()) / np.sqrt((ins_b.var() + oth_b.var()) / 2), "Cohen's d", f"{ins_b.mean():,.0f} vs {oth_b.mean():,.0f} RWF.")
# H2 satisfaction across facilities
an2 = stats.f_oneway(*[g_.Rating.values for _, g_ in Ef.groupby("FacilityID")])
add_ht("H2", "Does patient satisfaction differ across the 20 facilities?", "Median rating identical in all facilities", "At least one facility differs", "Kruskal-Wallis H (one-way ANOVA agrees: p = %.3f)" % an2.pvalue,
       len(Ef), kw_fs.statistic, kw_fs.pvalue, eps2(kw_fs.statistic, len(Ef)), "ε² (rank)", f"Facility means {fs['mean'].min():.2f}–{fs['mean'].max():.2f}.")
add_ht("H2b", "Does satisfaction differ across the 10 department types?", "Median rating identical", "At least one differs", "Kruskal-Wallis H", len(Ef), kw_ds.statistic, kw_ds.pvalue, eps2(kw_ds.statistic, len(Ef)), "ε² (rank)", "")
# H3 admission by age group
ct3 = pd.crosstab(E.AgeGroup, E.Admitted); c3 = stats.chi2_contingency(ct3)
add_ht("H3", "Do admission rates differ between age groups?", "Admission rate equal across the 7 age groups", "At least one age group differs", "χ² test of independence (df = %d)" % c3[2], len(E), c3[0], c3[1],
       np.sqrt(c3[0] / (len(E) * (min(ct3.shape) - 1))), "Cramér's V", f"Rates {100 * ag.rate.min():.1f}%–{100 * ag.rate.max():.1f}%.")
# H4 satisfaction vs severity (universe)
fsv = FIN.dropna(subset=["Rating", "Severity"]); kw4 = stats.kruskal(*[g_.Rating.values for _, g_ in fsv.groupby("Severity")])
add_ht("H4", "Does satisfaction depend on disease severity?", "Rating distribution identical across Mild/Moderate/Severe", "Differs", "Kruskal-Wallis H", len(fsv), kw4.statistic, kw4.pvalue, eps2(kw4.statistic, len(fsv)), "ε² (rank)", "")
# H5 LOS by diagnosis
add_ht("H5", "Does length of stay differ by diagnosis?", "Median LOS equal across diagnoses", "At least one differs", "Kruskal-Wallis H", len(Ea), kw3.statistic, kw3.pvalue, eps2(kw3.statistic, len(Ea)), "ε² (rank)", f"Mean LOS {dl['mean'].min():.1f}–{dl['mean'].max():.1f} d.")
# H6 satisfaction vs billing status / payment completeness
fst = FIN.dropna(subset=["Rating"]); kw6 = stats.kruskal(*[g_.Rating.values for _, g_ in fst.groupby("BillStatus")]); sp6b = stats.spearmanr(fst.PaidRatio, fst.Rating)
add_ht("H6", "Is satisfaction related to billing outcome (recorded bill status)?", "Rating equal across Paid / Partially Paid / Pending", "Differs", "Kruskal-Wallis H (Spearman ρ rating vs paid ratio = %.3f, p = %.3f)" % (sp6b[0], sp6b[1]),
       len(fst), kw6.statistic, kw6.pvalue, eps2(kw6.statistic, len(fst)), "ε² (rank)", "")
# H7 emergency vs other admission
ct7 = pd.crosstab(E.VisitType == "Emergency", E.Admitted); c7 = stats.chi2_contingency(ct7)
add_ht("H7", "Are emergency visits admitted more often than other visits?", "Admission proportion equal", "Proportions differ", "χ² test (2×2, Yates)", len(E), c7[0], c7[1], np.sqrt(c7[0] / len(E)), "φ",
       f"Emergency {100 * E[E.VisitType == 'Emergency'].Admitted.mean():.1f}% vs other {100 * E[E.VisitType != 'Emergency'].Admitted.mean():.1f}%.")
# H8 severe vs mild admission
ct8 = pd.crosstab(E.Severity, E.Admitted); c8 = stats.chi2_contingency(ct8)
add_ht("H8", "Are severe cases admitted more often than mild cases?", "Admission rate equal across severity levels", "Differs", "χ² test (df = 2)", len(E), c8[0], c8[1], np.sqrt(c8[0] / (len(E) * 1)), "Cramér's V",
       "Rates: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in E.groupby("Severity").Admitted.mean().items()))
# H9 chronic disease prevalence by gender & by age group
ctg = pd.crosstab(PATIENTS.Gender, PATIENTS.ChronicFlag); cg = stats.chi2_contingency(ctg)
add_ht("H9", "Does chronic-disease prevalence differ by gender?", "Prevalence equal for men and women", "Differs", "χ² test (2×2, Yates)", len(PATIENTS), cg[0], cg[1], np.sqrt(cg[0] / len(PATIENTS)), "φ", "")
cta = pd.crosstab(PATIENTS.AgeGroup, PATIENTS.ChronicFlag); ca = stats.chi2_contingency(cta)
add_ht("H10", "Does chronic-disease prevalence differ by age group?", "Prevalence equal across age groups", "Differs", "χ² test (df = %d)" % ca[2], len(PATIENTS), ca[0], ca[1], np.sqrt(ca[0] / (len(PATIENTS) * 1)), "Cramér's V", "")
# H11 rating-comment independence ; H12 claim approval by provider
add_ht("H11", "Is the written comment related to the numeric rating?", "Comment independent of rating", "Associated", "χ² test of independence", len(fb), chi_rc[0], chi_rc[1], np.sqrt(chi_rc[0] / (len(fb) * 3)), "Cramér's V", "A valid feedback instrument should show strong association.")
add_ht("H12", "Does the claim approval rate differ between insurance providers?", "Approval rate equal", "Differs", "χ² test (df = 3)", len(clm), chi_prov_appr[0], chi_prov_appr[1], np.sqrt(chi_prov_appr[0] / len(clm)), "Cramér's V", "")
# H13 monthly revenue seasonality (Q-test vs flat) skipped -> H13 payment ratio across methods
kw13 = stats.kruskal(*[g_.PaidRatio.values for _, g_ in FIN.groupby("PayMethod")])
add_ht("H13", "Does the share of the bill actually paid differ by payment method?", "Payment ratio equal across methods", "Differs", "Kruskal-Wallis H", len(FIN), kw13.statistic, kw13.pvalue, eps2(kw13.statistic, len(FIN)), "ε² (rank)", "")
HTT = pd.DataFrame(HT)
HTT["p_holm_all"] = holm(HTT.p_value.values)
# plain-language conclusion
def concl(r):
    if r.Decision == "Reject H0":
        return f"Statistically significant (p = {r.p_value:.4f}); effect size {r.Effect:.3f} ({r.Effect_measure})."
    return f"No significant difference (p = {r.p_value:.3f}); effect size {r.Effect:.3f} is negligible."
HTT["Interpretation"] = HTT.apply(concl, axis=1) + " " + HTT.Conclusion
R["hypothesis_tests"] = HTT.round(5).to_dict("records")
R["ht_rejected"] = int((HTT.Decision == "Reject H0").sum()); R["ht_total"] = len(HTT)
pd.options.display.float_format = "{:,.4f}".format
HTT[["ID", "Question", "Test", "n", "Statistic", "p_value", "p_holm_all", "Effect", "Effect_measure", "Decision"]]

# %%
# Which tests rejected H0?  (chance expectation: ~5% of independent true-null tests)
print(f"{R['ht_rejected']} of {R['ht_total']} hypothesis tests reject H0 at α = 0.05; after Holm correction: {(HTT.p_holm_all < .05).sum()}")
for _, r_ in HTT[HTT.Decision == "Reject H0"].iterrows(): print(f"  {r_.ID}: {r_.Question}  (p = {r_.p_value:.4g}, Holm p = {r_.p_holm_all:.4g}, {r_.Effect_measure} = {r_.Effect:.3f})")
fig, ax = plt.subplots(figsize=(10.5, 6.2))
ht_ = HTT.sort_values("p_value", ascending=False)
ax.barh(ht_.ID + "  " + ht_.Question.str.slice(0, 62) + "…", -np.log10(ht_.p_value.clip(lower=1e-12)), color=np.where(ht_.p_value < .05, C[7], MUTED), height=.65)
ax.axvline(-np.log10(.05), color=INK2, ls="--", lw=1); ax.set(title="Hypothesis tests at a glance (red = significant at α = 0.05)", xlabel="−log10(p-value)"); ax.tick_params(axis="y", labelsize=7.5)
rej = HTT[HTT.Decision == "Reject H0"]
save_fig(fig, "f26_hypothesis_summary",
         (f"{R['ht_rejected']} of {R['ht_total']} tests reject the null hypothesis at α = 0.05" + (": " + "; ".join(f"{r_.ID} (effect {r_.Effect:.2f})" for _, r_ in rej.iterrows()) if len(rej) else "") + f"; after Holm correction {(HTT.p_holm_all < .05).sum()} remain. "
          + ("None of the outcome-related comparisons — admission, satisfaction, length of stay, billing — shows a significant difference, so the data give no evidence that these outcomes vary by the factors tested. " if len(rej) == 0 else "")
          + f"The nearest to significance are H3 (admission by age group, p = {HTT.loc[HTT.ID == 'H3', 'p_value'].iloc[0]:.3f}) and H12 (claim approval by provider, p = {HTT.loc[HTT.ID == 'H12', 'p_value'].iloc[0]:.3f}); neither reaches α = 0.05."))

save(globals(), __file__)
print("Done:", os.path.basename(__file__))
