"""
HealthConnect Rwanda - Week 6 (Data Science track)
Part 6 - Business recommendations, strategic answers, export of datasets and tables

Run from the repository root:  python src/06_recommendations_and_export.py
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
# ## 7. Business recommendations
# Every recommendation cites the evidence (figure/table) that supports it **and** states how strong that evidence is. Where the data cannot support a conclusion
# the recommendation is to fix the data collection first — an evidence-based answer is sometimes "we cannot tell yet".

# %%
pend_claims = clm[clm.Status == "Pending"]; dis_share = float((fb.Rating <= 2).mean())
n_need = int(np.ceil(2 * (1.96 + .84) ** 2 * fb.Rating.var() / .3 ** 2))
ins_unreconciled = float(F[(F.Method == "Insurance") & (F.HasClaim == 0)].AmountPaid.sum())
fully_paid_share = float((F.PaidRatio == 1).mean())
hr = PATIENTS[(PATIENTS.HighRisk == 1)]; hr_seen = E[E.HighRisk == 1].PatientID.nunique()
fac_gap = K.Appointments.max() / K.Appointments.min() - 1
R["recs_evidence"] = dict(pending_claims_n=int(len(pend_claims)), pending_claims_value=float(pend_claims.ClaimAmount.sum()), dissatisfied_share=dis_share, n_per_facility_needed=n_need,
                          fully_paid_share=fully_paid_share, highrisk_n=int(len(hr)), highrisk_pct=float(len(hr) / len(PATIENTS)), highrisk_seen=int(hr_seen), fac_gap=float(fac_gap),
                          claim_approved_value=float(clm[clm.Status == "Approved"].ClaimAmount.sum()), mean_claim_ratio=float(F.ClaimRatio.mean()))
REC = pd.DataFrame([
 ["Operational challenge", "Fix the data foundation before further modelling",
  f"Only {R['n_encounters']:,} of 60,000 billed appointments exist in the Appointments table; 95% of appointments point to a department owned by another facility; no facility master, bed status or equipment availability fields.", "Figs. DQ table; §1.4", "High — verified by direct key checks",
  "Re-extract Appointments for the full year; enforce Facility→Department→Doctor foreign keys; add Facility master, bed status and equipment condition fields."],
 ["Operational challenge", "Stop using bill 'Status' as a collections indicator",
  f"{pct(fully_paid_share, 0)} of bills are paid in full whatever Status says; 'Pending' bills have the same average payment ratio ({100 * sp_.paid_ratio.mean():.0f}%) as 'Paid'.", "f14", "High",
  "Derive Status from Amount vs AmountPaid in the billing system; reconcile Revenue_Summary (1.13 bn) with billing (9.31 bn)."],
 ["High-risk patients", "Target outreach with a rule-based list, not a predictive model",
  f"{len(hr):,} patients ({pct(len(hr) / len(PATIENTS))}) meet the high-risk rule (65+/50–64 with chronic disease, obesity or underweight); only {hr_seen:,} had an appointment in 2026. Chronic disease affects {pct(PATIENTS.ChronicFlag.mean())} of patients but no demographic factor predicts it (AUC {auc_chr.mean():.2f}).", "f03, f24, f25, f33", "Medium — rule is clinical logic; data cannot validate it",
  "Call-back / screening programme for the high-risk list, prioritising patients aged 65+ with a chronic condition and no 2026 visit; evaluate uptake to validate the rule."],
 ["Patient satisfaction", "Redesign the feedback instrument and capture real drivers",
  f"Mean rating {fb.Rating.mean():.2f}/5; {pct(dis_share, 0)} rate 1–2. Ratings are unrelated to comment text, diagnosis, billing, department, facility or any modelled variable (best R² ≈ 0).", "f16, f17, f32", "High (that data cannot explain satisfaction); drivers unknown",
  f"Add wait time, staff courtesy, cleanliness, drug availability and discharge-time questions; validate comment/rating consistency; collect ≥ {n_need} ratings per facility to detect a 0.3-point difference with 80% power (currently ≈ {int(K.RatedN.median())})."],
 ["Patient satisfaction", "Do not rank facilities or departments on current ratings",
  f"Facility means {fs['mean'].min():.2f}–{fs['mean'].max():.2f}, department means {ds['mean'].min():.2f}–{ds['mean'].max():.2f}; Kruskal-Wallis p = {kw_fs.pvalue:.2f} (facilities), {kw_ds.pvalue:.2f} (departments).", "f17, H2", "High",
  "Publish confidence intervals with any league table; investigate the lowest-rated sites only after the new instrument yields adequate samples."],
 ["Resource optimisation", "Make staffing follow patient volume",
  f"Facility volumes range {int(K.Appointments.min())}–{int(K.Appointments.max())} ({pct(fac_gap, 0)} spread) while headcount is {int(K.TotalStaff.min())}–{int(K.TotalStaff.max())} everywhere; staff numbers do not track volume (ρ = {rr[0]:.2f}). Doctor load is random (variance/mean {disp:.2f}), i.e. nobody is systematically overloaded.", "f10, f11", "Medium — the volume gap is not statistically significant (p = 0.052) but is practically large",
  "Move toward a staffing ratio per 1,000 appointments; rebalance rotas from the quietest sites toward HF013/HF011 as a pilot; review monthly with a rolling 3-month workload index."],
 ["Resource optimisation", "Verify bed capacity before any bed-related investment",
  f"All {R['beds_total']:,} beds are flagged 'Available'; recorded admissions imply only ≈{100 * R['bed_occupancy_system']:.1f}% annual bed-day use.", "f12", "Low — Admissions table appears to be an extract",
  "Capture daily bed census; if the 4% figure holds, consolidate wards rather than expand."],
 ["Resource optimisation", "Review lab-test ordering",
  f"Lab test type is unrelated to diagnosis (χ² {fmt_p(R['lab_dx_p'])}) — e.g. malaria patients are not more likely to get a malaria test — while prescribing matches diagnosis {pct(R['rx_concordance'], 0)}.", "f08", "Medium — may be a data-linkage artefact",
  "Audit test-ordering against diagnostic protocols; link lab orders to diagnosis codes."],
 ["Revenue improvement", "Collect the outstanding balance and clear pending claims",
  f"{R['outstanding_total'] / 1e9:.2f} bn RWF ({pct(R['outstanding_pct'])}) of billed amounts is unpaid; {len(pend_claims):,} claims worth {pend_claims.ClaimAmount.sum() / 1e9:.2f} bn RWF are 'Pending' and only {pct(R['claim_approval'], 0)} of claims are approved (no difference by provider, p = {R['provider_approval_p']:.2f}).", "f14", "High",
  "Weekly ageing report on outstanding balances; dedicated follow-up team for pending claims; agree turnaround SLAs with all four insurers; {} RWF of 'Insurance'-method payments have no claim on file — reconcile.".format(f"{ins_unreconciled / 1e9:.2f} bn")],
 ["Revenue improvement", "Budget on a flat baseline and manage volatility",
  f"2027 forecast ≈ {fc_rev.Forecast.sum() / 1e9:.2f} bn RWF ({fc_rev.Forecast.mean() / 1e6:.0f} M/month, 95% band ±{100 * w95r / fc_rev.Forecast.mean():.0f}%); no trend is detectable (12 months cannot reveal seasonality), and revenue is only weakly tied to visit counts (r = {R['rev_appt_corr'][0]:.2f}).", "f27, f28", "Medium — only 12 monthly points",
  "Set the budget at the forecast mean with a contingency reserve; analyse the Sep-2026 peak (146 M) and Dec-2026 trough (67 M) for one-off billing events; collect 24+ months before using seasonal models."],
 ["Service expansion", "Do not expand on current evidence — build the evidence",
  f"Demand is flat (no trend, ρ = {diag_app['trend_rho']:.2f}), evenly spread across provinces ({pct(prov.Patients.min() / prov.Patients.sum())}–{pct(prov.Patients.max() / prov.Patients.sum())}) and departments; admissions cannot be predicted; facility locations and service-level data are missing.", "f05, f27, f29", "Medium",
  "Add facility geography (district, ownership, type) and ServiceID on bills; re-run province-level demand vs capacity once available. Prioritise Cardiology and Radiology for a staffing review (highest forecast load index, ≈105–109), but treat as provisional."],
], columns=["Theme", "Recommendation", "Evidence", "Where in analysis", "Strength of evidence", "Suggested action"])
R["recommendations"] = REC.to_dict("records")
pd.set_option("display.max_colwidth", 140)
REC[["Theme", "Recommendation", "Strength of evidence"]]

# %% [markdown]
# ## 8. Answers to the strategic questions
# | Question | Answer from the analysis |
# |---|---|
# | Which factors influence hospital admissions? | **None can be confirmed.** 26 factors screened; none significant after correction; best model AUC ≈ 0.51 (chance). |
# | Which patient groups are most at risk of chronic disease? | Prevalence is 75% everywhere; only blood group differs significantly (effect negligible); demographic model AUC 0.51. Use the clinical risk rule (65+ with chronic disease/obesity) for outreach. |
# | What factors affect patient satisfaction? | **None identifiable** in the recorded data (R² ≈ 0). The feedback instrument itself looks unreliable. |
# | Can future patient demand be forecast accurately? | Only as a stable average (~5,000 visits/month, ±14% typical error); no trend in 12 months (seasonality cannot be assessed with so little history). |
# | Which departments are likely to become overloaded? | None clearly. Cardiology and Radiology show the highest forecast load index (≈109 and 105) but differences are not statistically significant. |
# | How can resources be allocated more efficiently? | Align staffing with facility volume (28% spread vs near-identical headcount); capture bed census before changing capacity. |
# | What operational improvements will maximise outcomes? | Fix data integrity first, then collections (0.92 bn unpaid), claims turnaround, lab-ordering audit and a redesigned feedback instrument. |

# %% [markdown]
# ## 9. Export — cleaned datasets, tables and results
# Direct identifiers (names, phone numbers, emergency contacts) are **not** included in any exported file.

# %%
TAB = OUT + "tables/"; os.makedirs(TAB, exist_ok=True); os.makedirs(OUT + "data/", exist_ok=True)
enc_cols = [c for c in E.columns if c not in ["Dept_Facility"]]
E.sort_values("AppointmentID").to_csv(OUT + "data/HC_Integrated_Encounters_5000.csv", index=False, date_format="%Y-%m-%d")
FIN.sort_values("BillingID").to_csv(OUT + "data/HC_Integrated_Billing_Feedback_60000.csv", index=False, date_format="%Y-%m-%d")
PATIENTS.to_csv(OUT + "data/HC_Clean_Patients_20000.csv", index=False, date_format="%Y-%m-%d")
STAFF.drop(columns=["FirstName", "LastName"], errors="ignore").to_csv(OUT + "data/HC_Clean_Staff_1250.csv", index=False)
DQT.to_csv(TAB + "data_quality_findings.csv", index=False); REL.to_csv(TAB + "relationship_checks.csv", index=False); DESC.to_csv(TAB + "descriptive_statistics.csv", index=False)
CORRT.to_csv(TAB + "correlations.csv", index=False); HTT.to_csv(TAB + "hypothesis_tests.csv", index=False); FT.to_csv(TAB + "admission_factor_scan.csv", index=False)
PREV.to_csv(TAB + "chronic_group_prevalence.csv", index=False); ADM.to_csv(TAB + "admission_model_cv.csv", index=False); SAT.to_csv(TAB + "satisfaction_model_cv.csv", index=False)
pd.concat([fc_app.assign(Series="Monthly visits"), fc_rev.assign(Series="Monthly revenue (RWF)")]).to_csv(TAB + "forecasts_2027.csv", index=False)
DF.to_csv(TAB + "department_workload_forecast.csv"); K.to_csv(TAB + "facility_kpis.csv"); DEPTT.to_csv(TAB + "department_type_kpis.csv"); REC.to_csv(TAB + "recommendations.csv", index=False)
with pd.ExcelWriter(OUT + "HC_Analysis_Tables.xlsx") as xw:
    for nm, df_ in [("Data quality", DQT), ("Relationships", REL), ("Descriptives", DESC), ("Correlations", CORRT), ("Hypothesis tests", HTT), ("Admission factors", FT), ("Chronic groups", PREV),
                    ("Admission models", ADM), ("Satisfaction models", SAT), ("Forecast visits", fc_app), ("Forecast revenue", fc_rev), ("Dept forecast", DF.reset_index()),
                    ("Facility KPIs", K.reset_index()), ("Dept type KPIs", DEPTT.reset_index()), ("Recommendations", REC)]:
        df_.to_excel(xw, sheet_name=nm[:31], index=False)
print("Exported:", sorted(os.listdir(OUT + "data")), "| tables:", len(os.listdir(TAB)), "| figures:", len(FIGS))
R["n_figures"] = len(FIGS)


# --- save headline numbers for the report/slide builders (07, 08) ---------------------------------
import json
def _j(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    return str(o)
json.dump({"R": R, "FIGS": FIGS, "DQ": DQ}, open(OUT + "results.json", "w"), indent=1, default=_j)
print("Saved", OUT + "results.json")

save(globals(), __file__)
print("Done:", os.path.basename(__file__))
