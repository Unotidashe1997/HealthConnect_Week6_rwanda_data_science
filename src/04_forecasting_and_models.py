"""
HealthConnect Rwanda - Week 6 (Data Science track)
Part 4 - Visit/revenue/workload forecasts, admission and satisfaction models (figures 27-32)

Run from the repository root:  python src/04_forecasting_and_models.py
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
# ## 5. Predictive analytics & forecasting
# Method notes that apply to every model below:
# * Models are judged on **out-of-sample** performance only (rolling-origin back-tests for time series; repeated stratified k-fold
#   cross-validation for classifiers/regressors) and always against a **naïve baseline** — a model that cannot beat "predict the average" has no value.
# * A **label-permutation test** checks whether any apparent skill exceeds chance.
# * Forecast horizon: the data cover Jan–Dec 2026, so "the next 12 months" is **Jan–Dec 2027**.
# * statsmodels is unavailable in this environment, so the exponential-smoothing models are implemented directly in NumPy (simple, Holt damped-trend)
#   alongside mean, naïve and linear-trend benchmarks. With only 12 monthly observations, seasonal models (which need ≥ 24) cannot be estimated.
#
# ### 5.1 Time-series tools

# %%
from itertools import product

def f_mean(y, h):  return np.repeat(np.mean(y), h)
def f_naive(y, h): return np.repeat(y[-1], h)
def f_trend(y, h):
    t = np.arange(len(y)); b = np.polyfit(t, y, 1); return np.polyval(b, np.arange(len(y), len(y) + h))
def _ses(y, a):
    l = y[0]; sse = 0.0
    for v in y[1:]:
        sse += (v - l) ** 2; l = a * v + (1 - a) * l
    return l, sse
def f_ses(y, h):
    best = min(((_ses(y, a)[1], a) for a in np.linspace(.05, 1, 20)))[1]; return np.repeat(_ses(y, best)[0], h)
def _holt(y, a, b, phi):
    l, tr, sse = y[0], y[1] - y[0], 0.0
    for v in y[1:]:
        f = l + phi * tr; sse += (v - f) ** 2; ln = a * v + (1 - a) * f; tr = b * (ln - l) + (1 - b) * phi * tr; l = ln
    return l, tr, sse
def f_holt(y, h):
    best = min(((_holt(y, a, b, p)[2], a, b, p) for a, b, p in product(np.linspace(.1, 1, 10), np.linspace(.05, .5, 6), [.8, .9, .98])))
    l, tr, _ = _holt(y, *best[1:]); return np.array([l + sum(best[3] ** k for k in range(1, i + 1)) * tr for i in range(1, h + 1)])

MODELS = {"Mean": f_mean, "Naïve (last value)": f_naive, "Linear trend": f_trend, "Simple exp. smoothing": f_ses, "Holt damped trend": f_holt}

def backtest(y, min_train=6, hmax=3):
    y = np.asarray(y, float); rows = []
    for name, fn in MODELS.items():
        errs, perr = [], []
        for t in range(min_train, len(y)):
            h = min(hmax, len(y) - t); f = fn(y[:t], h); a = y[t:t + h]
            errs += list(a - f); perr += list(np.abs(a - f) / np.abs(a))
        errs = np.array(errs); rows.append(dict(Model=name, MAE=np.mean(np.abs(errs)), RMSE=np.sqrt(np.mean(errs ** 2)), MAPE=100 * np.mean(perr), Folds=len(errs)))
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)

def forecast_series(y, dates, h=12):
    bt = backtest(y); best = bt.iloc[0].Model; base_mae = bt.set_index("Model").loc["Mean", "MAE"]
    fc = MODELS[best](np.asarray(y, float), h); sig = bt.iloc[0].RMSE
    fd = pd.date_range(dates.iloc[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    return bt, best, pd.DataFrame({"Month": fd, "Forecast": fc, "Lower95": fc - 1.96 * sig, "Upper95": fc + 1.96 * sig}), sig

def trend_diag(y):
    y = np.asarray(y, float); t = np.arange(len(y)); sp = stats.spearmanr(t, y); y0 = y - y.mean()
    ac1 = (y0[1:] * y0[:-1]).sum() / (y0 ** 2).sum(); slope = np.polyfit(t, y, 1)[0]
    return dict(trend_rho=sp[0], trend_p=sp[1], lag1_acf=ac1, acf_bound=1.96 / np.sqrt(len(y)), slope_pct_per_month=100 * slope / y.mean())

# %% [markdown]
# ### 5.2 Task 1 — Patient-volume forecast (next 12 months)
# Series used: the **Revenue_Summary monthly appointment counts** (system-wide, ~60,800 appointments), cross-checked with the monthly counts in the
# 5,000-appointment record. The rolling-origin back-test compares five models; the best by MAE produces the forecast, and 95% bands use the
# back-test RMSE (so the width reflects real out-of-sample error rather than a theoretical formula).

# %%
y_app = rs.Appointments.values.astype(float)
bt_app, best_app, fc_app, sig_app = forecast_series(y_app, rs.Month)
diag_app = trend_diag(y_app)
y_cohort = E.groupby("Month").size().values.astype(float); bt_coh, best_coh, fc_coh, _ = forecast_series(y_cohort, pd.Series(sorted(E.Month.unique())))
diag_coh = trend_diag(y_cohort)
mean_mape_app = bt_app.set_index("Model").loc["Mean", "MAPE"]
R["fc_app"] = dict(best=best_app, backtest=bt_app.round(2).to_dict("records"), mape_best=float(bt_app.iloc[0].MAPE), mape_mean=float(mean_mape_app), diag=diag_app,
                   forecast_total_2027=float(fc_app.Forecast.sum()), forecast_monthly=fc_app.Forecast.round(0).tolist(), lower=fc_app.Lower95.round(0).tolist(), upper=fc_app.Upper95.round(0).tolist(), history_total_2026=float(y_app.sum()))
R["fc_app_cohort"] = dict(best=best_coh, mape_best=float(bt_coh.iloc[0].MAPE), diag=diag_coh)
pd.options.display.float_format = "{:,.2f}".format
print("Back-test (rolling origin, 1–3 month horizon) — system appointments:"); print(bt_app.to_string(index=False))
print("\nTrend/autocorrelation diagnostics:", {k: round(v, 3) for k, v in diag_app.items()})
fc_app.round(0)

# %%
def fc_plot(ax, dates, y, fc, title, ylabel, scale=1, color=C[0]):
    ax.plot(dates, y / scale, color=color, lw=2, marker="o", ms=4, label="2026 actual")
    ax.plot(fc.Month, fc.Forecast / scale, color=C[1], lw=2, marker="o", ms=4, label="2027 forecast")
    ax.fill_between(fc.Month, fc.Lower95 / scale, fc.Upper95 / scale, color=C[1], alpha=.18, label="95% band (back-test error)")
    ax.axvline(pd.Timestamp("2026-12-16"), color=MUTED, ls=":", lw=1); ax.set(title=title, xlabel="Month", ylabel=ylabel); ax.legend(loc="upper left", fontsize=8)
    for t_ in ax.get_xticklabels(): t_.set_rotation(40)

fig, ax = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
fc_plot(ax[0], rs.Month, y_app, fc_app, f"Monthly patient visits — 12-month forecast ({best_app})", "Appointments per month")
b_ = bt_app.sort_values("MAPE"); ax[1].barh(b_.Model[::-1], b_.MAPE[::-1], color=[C[1] if m == best_app else MUTED for m in b_.Model[::-1]], height=.6)
for i, v in enumerate(b_.MAPE[::-1]): ax[1].text(v + .3, i, f"{v:.1f}%", va="center", fontsize=8.5)
ax[1].set(title="Back-test error by model (MAPE, lower is better)", xlabel="Mean absolute % error", xlim=(0, b_.MAPE.max() * 1.15))
fig.tight_layout()
w95 = (fc_app.Upper95 - fc_app.Lower95).mean() / 2
save_fig(fig, "f27_forecast_visits",
         f"Monthly visits fluctuate around {y_app.mean():,.0f} with no trend (Spearman ρ = {diag_app['trend_rho']:.2f}, {fmt_p(diag_app['trend_p'])}) and no significant month-to-month autocorrelation (lag-1 = {diag_app['lag1_acf']:.2f}, inside the ±{diag_app['acf_bound']:.2f} noise band). "
         f"The best back-tested model is '{best_app}' — only marginally better than the plain average (MAPE {bt_app.iloc[0].MAPE:.1f}% vs {mean_mape_app:.1f}%) — and forecasts ≈{fc_app.Forecast.mean():,.0f} visits a month for 2027 ({fc_app.Forecast.sum():,.0f} for the year, {pct(fc_app.Forecast.sum() / y_app.sum() - 1)} versus 2026). "
         f"Honest answer to 'can demand be forecast accurately?': only at the level of a stable average — typical monthly error is ±{bt_app.iloc[0].MAPE:.0f}% and the 95% band is ±{w95:,.0f} visits ({100 * w95 / fc_app.Forecast.mean():.0f}%). A real seasonal model needs at least 24 months of history.")

# %% [markdown]
# ### 5.3 Task 2 — Revenue forecast (next 12 months)

# %%
y_rev = rs.Revenue.values.astype(float)
bt_rev, best_rev, fc_rev, sig_rev = forecast_series(y_rev, rs.Month)
diag_rev = trend_diag(y_rev)
R["fc_rev"] = dict(best=best_rev, backtest=bt_rev.round(2).to_dict("records"), mape_best=float(bt_rev.iloc[0].MAPE), mape_mean=float(bt_rev.set_index("Model").loc["Mean", "MAPE"]), diag=diag_rev,
                   forecast_total_2027=float(fc_rev.Forecast.sum()), history_total_2026=float(y_rev.sum()), forecast_monthly=fc_rev.Forecast.round(0).tolist(), lower=fc_rev.Lower95.round(0).tolist(), upper=fc_rev.Upper95.round(0).tolist())
# revenue = visits × revenue-per-visit: decompose the uncertainty
print(bt_rev.to_string(index=False)); print("\nDiagnostics:", {k: round(v, 3) for k, v in diag_rev.items()})
fig, ax = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
fc_plot(ax[0], rs.Month, y_rev, fc_rev, f"Monthly revenue — 12-month forecast ({best_rev})", "Revenue (million RWF)", scale=1e6, color=C[0])
b_ = bt_rev.sort_values("MAPE"); ax[1].barh(b_.Model[::-1], b_.MAPE[::-1], color=[C[1] if m == best_rev else MUTED for m in b_.Model[::-1]], height=.6)
for i, v in enumerate(b_.MAPE[::-1]): ax[1].text(v + .3, i, f"{v:.1f}%", va="center", fontsize=8.5)
ax[1].set(title="Back-test error by model (MAPE, lower is better)", xlabel="Mean absolute % error", xlim=(0, b_.MAPE.max() * 1.15))
fig.tight_layout()
w95r = (fc_rev.Upper95 - fc_rev.Lower95).mean() / 2
save_fig(fig, "f28_forecast_revenue",
         f"Revenue shows no detectable trend (Spearman ρ = {diag_rev['trend_rho']:.2f}, {fmt_p(diag_rev['trend_p'])}; lag-1 autocorrelation {diag_rev['lag1_acf']:.2f}). The best back-tested model is '{best_rev}' (MAPE {bt_rev.iloc[0].MAPE:.0f}%), giving "
         f"≈{fc_rev.Forecast.mean() / 1e6:,.0f} M RWF per month and {fc_rev.Forecast.sum() / 1e9:.2f} bn RWF for 2027 ({pct(fc_rev.Forecast.sum() / y_rev.sum() - 1)} vs 2026's {y_rev.sum() / 1e9:.2f} bn). "
         f"Uncertainty is large — the 95% band is ±{w95r / 1e6:,.0f} M per month (±{100 * w95r / fc_rev.Forecast.mean():.0f}%) — because single months such as Sep-2026 ({rs.Revenue.max() / 1e6:.0f} M) and Dec-2026 ({rs.Revenue.min() / 1e6:.0f} M) are far from the average. Use the 2027 figure as a budget baseline, not a monthly target.")

# %% [markdown]
# ### 5.4 Task 5 — Department workload forecast
# Department demand in the appointment record is too sparse to forecast per department-ID (≈2 appointments per department per month), so
# demand is forecast at **department-type** level: the system-wide visit forecast (5.2) is split by each type's observed share of appointments, with
# share uncertainty (Dirichlet) and forecast error (normal, back-test RMSE) propagated by Monte-Carlo simulation (5,000 runs). Capacity is the number
# of clinicians (doctors + nurses) assigned to the department type. A department-type is **overload-prone** when simulated monthly appointments per clinician exceed 115 % of the all-department average.

# %%
cnt = E.groupby("DepartmentName").size(); share = cnt / cnt.sum()
clin = STAFF[STAFF.Clinician == 1].groupby("Department").size().reindex(cnt.index)
sim = 5000; rng2 = np.random.default_rng(SEED)
tot_draw = rng2.normal(fc_app.Forecast.values[None, :], sig_app, size=(sim, 12)).clip(min=0)          # sim × 12 system visits
sh_draw = rng2.dirichlet(cnt.values + 1, size=sim)                                                      # sim × 10 shares
dem = tot_draw[:, :, None] * sh_draw[:, None, :]                                                         # sim × month × dept
load = dem / clin.values[None, None, :]                                                                  # appointments per clinician per month
thr = 1.15 * (fc_app.Forecast.mean() / clin.sum())
p_over = (load > thr).mean(axis=(0, 1))
DF = pd.DataFrame({"Share_of_visits": share, "Clinicians": clin, "Fcst_visits_per_month": share * fc_app.Forecast.mean(), "Fcst_per_clinician": share * fc_app.Forecast.mean() / clin,
                   "P_overload": p_over, "Hist_WorkloadIndex": DEPTT.WorkloadIndex}).sort_values("P_overload", ascending=False)
DF["Fcst_WorkloadIndex"] = 100 * DF.Fcst_per_clinician / (fc_app.Forecast.mean() / clin.sum())
chi_sh = stats.chisquare(cnt.values)
R["dept_forecast"] = DF.round(4).reset_index().rename(columns={"index": "Department"}).to_dict("records"); R["dept_share_p"] = float(chi_sh.pvalue)
pd.options.display.float_format = "{:,.3f}".format
DF

# %%
# seasonal pattern check by department (is there a month effect at all?)
dm = pd.crosstab(E.DepartmentName, E.MonthNum); chi_dm = stats.chi2_contingency(dm)
fig, ax = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1.1, 1.1, 1.4]})
d1 = DF.sort_values("Fcst_visits_per_month")
ax[0].barh(d1.index, d1.Fcst_visits_per_month, color=C[0], height=.65)
for i, v in enumerate(d1.Fcst_visits_per_month): ax[0].text(v + 4, i, f"{v:,.0f}", va="center", fontsize=8.5)
ax[0].set(title="Forecast 2027 visits per month by department type", xlabel="Appointments per month", xlim=(0, d1.Fcst_visits_per_month.max() * 1.15))
d2 = DF.sort_values("Fcst_WorkloadIndex")
ax[1].barh(d2.index, d2.Fcst_WorkloadIndex, color=np.where(d2.P_overload > .5, C[7], np.where(d2.P_overload > .25, C[3], C[2])), height=.65); ax[1].axvline(100, color=INK2, ls="--", lw=1); ax[1].axvline(115, color=C[7], ls=":", lw=1)
for i, (v, p_) in enumerate(zip(d2.Fcst_WorkloadIndex, d2.P_overload)): ax[1].text(v + 1, i, f"{v:.0f}  (P={p_:.0%})", va="center", fontsize=8.5)
ax[1].set(title="Forecast workload index (100 = average; dotted = 115 overload line)", xlabel="Appointments per clinician, indexed", xlim=(60, 160)); ax[1].set_yticklabels([])
for j, dname in enumerate(DF.index[:4]):
    ax[2].plot(range(1, 13), dm.loc[dname].values, marker="o", ms=3.5, lw=1.8, color=C[j], label=dname)
ax[2].set(title="Monthly appointments, four busiest-per-clinician departments (2026)", xlabel="Month", ylabel="Appointments"); ax[2].legend(fontsize=8)
fig.tight_layout()
top_dep = DF.index[0]
save_fig(fig, "f29_department_workload_forecast",
         f"Demand is spread almost evenly over the ten department types ({100 * share.min():.1f}%–{100 * share.max():.1f}% of visits; χ² for equal shares {fmt_p(chi_sh.pvalue)}) and shows no month-of-year pattern by department (χ² {fmt_p(chi_dm[1])}). "
         f"With clinician numbers also equal, forecast workload indices lie between {DF.Fcst_WorkloadIndex.min():.0f} and {DF.Fcst_WorkloadIndex.max():.0f}; the highest is {top_dep} (index {DF.Fcst_WorkloadIndex.iloc[0]:.0f}, simulated probability of exceeding the 115 line {DF.P_overload.iloc[0]:.0%}). "
         f"Differences between types are not statistically significant, so no department type is clearly headed for overload; the risk is spread thinly and is dominated by random variation in monthly demand, so staffing decisions should rely on monitoring (rolling 3-month load) rather than a fixed forecast.")

# %% [markdown]
# ### 5.5 Task 3 — Hospital-admission prediction
# **Target:** the appointment resulted in an admission (586 of 5,000, 11.7 %). **Feature sets** are nested so that we can see what each information layer adds:
# * **A — at registration/booking:** demographics, chronic condition, BMI, risk score, visit type, department, facility, month, weekday, doctor experience, prior visits
# * **B — + clinical assessment:** diagnosis and severity
# * **C — + investigations:** lab done/abnormal, medication, referral (available only after the visit — shown for completeness, not for use in advance triage)
#
# Models: majority-class baseline, regularised logistic regression, random forest, histogram gradient boosting. 5-fold × 3-repeat stratified CV.

# %%
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold, cross_validate, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, r2_score, mean_absolute_error, roc_curve
from sklearn.inspection import permutation_importance

Ew = E.copy(); Ew["LabAbnormalNum"] = Ew.LabAbnormalNum.fillna(0); Ew["Medication"] = Ew.Medication.fillna("None"); Ew["Month_"] = Ew.MonthNum
FS_A_num = ["Age", "BMI", "RiskScore", "DoctorYearsExp", "PatientVisits", "Month_", "ChronicFlag"]
FS_A_cat = ["Gender", "MaritalStatus", "Education", "Occupation", "ProvinceName", "BloodGroup", "ChronicCondition", "VisitType", "DepartmentName", "FacilityID", "DayOfWeek"]
FS_B_cat = FS_A_cat + ["Diagnosis", "Severity"]
FS_C_num = FS_A_num + ["LabDone", "LabAbnormalNum", "Referred"]; FS_C_cat = FS_B_cat + ["Medication"]
SETS = {"A  Booking-time": (FS_A_num, FS_A_cat), "B  + Diagnosis & severity": (FS_A_num, FS_B_cat), "C  + Labs, Rx, referral": (FS_C_num, FS_C_cat)}

def make_pipe(num, cat, est):
    return Pipeline([("p", ColumnTransformer([("n", StandardScaler(), num), ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), cat)])), ("m", est)])

CLFS = {"Baseline (prior)": lambda: DummyClassifier(strategy="prior"),
        "Logistic regression": lambda: LogisticRegression(C=.1, max_iter=2000),
        "Random forest": lambda: RandomForestClassifier(300, min_samples_leaf=25, max_features="sqrt", n_jobs=-1, random_state=SEED),
        "Gradient boosting": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=.05, max_iter=120, l2_regularization=5, random_state=SEED)}
yA = Ew.Admitted.values; cvA = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
res = []
for sname, (num, cat) in SETS.items():
    X = Ew[num + cat]
    for mname, mk in CLFS.items():
        cvr = cross_validate(make_pipe(num, cat, mk()), X, yA, cv=cvA, scoring={"auc": "roc_auc", "ap": "average_precision", "brier": "neg_brier_score"}, n_jobs=1)
        res.append(dict(FeatureSet=sname, Model=mname, AUC=cvr["test_auc"].mean(), AUC_sd=cvr["test_auc"].std(), PR_AUC=cvr["test_ap"].mean(), Brier=-cvr["test_brier"].mean()))
ADM = pd.DataFrame(res); prev = yA.mean(); ADM["Brier_baseline"] = prev * (1 - prev)
R["admission_models"] = ADM.round(4).to_dict("records"); R["admission_prevalence"] = float(prev)
pd.options.display.float_format = "{:,.3f}".format
ADM

# %%
# permutation test: is the best CV AUC distinguishable from chance? (logistic regression, feature set B, 100 label shuffles)
numB, catB = SETS["B  + Diagnosis & severity"]; XB = Ew[numB + catB]
obs = ADM[(ADM.FeatureSet == "B  + Diagnosis & severity") & (ADM.Model == "Logistic regression")].AUC.iloc[0]
cv1 = StratifiedKFold(5, shuffle=True, random_state=SEED)
null_auc = []
for i in range(100):
    ysh = rng.permutation(yA)
    null_auc.append(cross_val_score(make_pipe(numB, catB, LogisticRegression(C=.1, max_iter=1000)), XB, ysh, cv=cv1, scoring="roc_auc").mean())
null_auc = np.array(null_auc); perm_p = (1 + (null_auc >= obs).sum()) / (1 + len(null_auc))
best_row = ADM.sort_values("AUC", ascending=False).iloc[0]
R["admission_perm_p"] = float(perm_p); R["admission_best"] = best_row.to_dict()

# hold-out evaluation of the best model, calibration & importance
Xtr, Xte, ytr, yte = train_test_split(Ew[SETS[best_row.FeatureSet][0] + SETS[best_row.FeatureSet][1]], yA, test_size=.2, stratify=yA, random_state=SEED)
num_b, cat_b = SETS[best_row.FeatureSet]
best_pipe = make_pipe(num_b, cat_b, CLFS[best_row.Model]()).fit(Xtr, ytr); pte = best_pipe.predict_proba(Xte)[:, 1]
auc_te = roc_auc_score(yte, pte)
bs_auc = [roc_auc_score(yte[ix], pte[ix]) for ix in (rng.integers(0, len(yte), len(yte)) for _ in range(500)) if len(np.unique(yte[ix])) == 2]
R["admission_holdout"] = dict(auc=float(auc_te), lo=float(np.percentile(bs_auc, 2.5)), hi=float(np.percentile(bs_auc, 97.5)), n_test=len(yte), pos_test=int(yte.sum()))
pi = permutation_importance(best_pipe, Xte, yte, scoring="roc_auc", n_repeats=15, random_state=SEED)
IMP = pd.DataFrame({"Feature": Xte.columns, "Importance": pi.importances_mean, "sd": pi.importances_std}).sort_values("Importance", ascending=False).head(10)

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
pv_ = ADM.pivot(index="Model", columns="FeatureSet", values="AUC").loc[["Baseline (prior)", "Logistic regression", "Random forest", "Gradient boosting"]]
w_ = .26
for j, fsn in enumerate(pv_.columns):
    ax[0].bar(np.arange(len(pv_)) + (j - 1) * w_, pv_[fsn], width=w_ - .02, color=C[j], label=fsn)
ax[0].axhline(.5, color=INK2, ls="--", lw=1); ax[0].set_xticks(range(len(pv_))); ax[0].set_xticklabels([m.replace(" ", "\n", 1) for m in pv_.index], fontsize=8)
ax[0].set(title="Cross-validated AUC by model and feature set", ylabel="ROC-AUC (0.5 = chance)", ylim=(.4, .65)); ax[0].legend(fontsize=7.5, loc="upper left")
ax[1].hist(null_auc, bins=18, color=MUTED, edgecolor=SURF); ax[1].axvline(obs, color=C[7], lw=2); ax[1].text(obs, ax[1].get_ylim()[1] * .9, f" observed {obs:.3f}\n permutation {fmt_p(perm_p)}", color=C[7], fontsize=8.5, va="top")
ax[1].set(title="Is the model better than chance? (100 label shuffles)", xlabel="AUC with shuffled labels", ylabel="Count")
dec = pd.DataFrame({"y": yte, "p": pte}); dec["bin"] = pd.qcut(dec.p.rank(method="first"), 5, labels=False)
cal = dec.groupby("bin").agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
ax[2].plot([0, .25], [0, .25], color=INK2, ls="--", lw=1); ax[2].plot(cal.pred, cal.obs, "o-", color=C[0], lw=2); ax[2].set(title="Calibration on hold-out (quintiles of predicted risk)", xlabel="Mean predicted probability", ylabel="Observed admission rate", xlim=(0, .25), ylim=(0, .25))
fig.tight_layout()
top_imp = IMP.iloc[0]
save_fig(fig, "f30_admission_model",
         f"Best cross-validated AUC is {best_row.AUC:.3f} ({best_row.Model}, feature set {best_row.FeatureSet.split()[0]}) — at or barely above chance (0.5); the label-shuffling test gives {fmt_p(perm_p)}, so {'the skill is statistically detectable but practically negligible' if perm_p < .05 else 'it is indistinguishable from chance'}. "
         f"Hold-out AUC {auc_te:.3f} (95% CI {np.percentile(bs_auc, 2.5):.2f}–{np.percentile(bs_auc, 97.5):.2f}). Adding diagnosis, severity, labs, medication and referral does not improve on booking-time information. "
         f"Calibration: predicted risks cluster near the {pct(prev)} base rate; observed admission is {100 * cal.obs.iloc[-1]:.1f}% in the top-risk quintile and {100 * cal.obs.iloc[0]:.1f}% in the lowest, a gap that is within sampling noise for only {int(cal.n.iloc[0])} patients per quintile — no usable risk stratification. Admissions in this data cannot be predicted from the recorded variables.")

# %%
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.barh(IMP.Feature[::-1], IMP.Importance[::-1], xerr=IMP["sd"][::-1], color=C[0], height=.6, ecolor=MUTED)
ax.axvline(0, color=INK2, lw=1); ax.set(title="Permutation importance — admission model (drop in hold-out AUC when feature is shuffled)", xlabel="Mean AUC decrease")
save_fig(fig, "f31_admission_importance", f"Even the most 'important' feature ({top_imp.Feature}) changes hold-out AUC by only {top_imp.Importance:.3f}, and most bars straddle zero (error bars = SD over 15 shuffles). "
         "Importance values this small indicate noise-fitting rather than genuine drivers, matching the factor screen in section 4.3.")
print("Admission probability scoring: predicted probability range on hold-out = %.3f – %.3f (base rate %.3f)" % (pte.min(), pte.max(), prev))

# %% [markdown]
# ### 5.6 Task 4 — Patient-satisfaction prediction
# **Target:** feedback rating (1–5). Two scopes, because information is available for different numbers of encounters:
# * **Scope 1 — all feedback (n = 21,064):** clinical & financial variables only (diagnosis, severity, bill amount, payment ratio/status/method, claim provider/status/coverage).
# * **Scope 2 — appointment cohort (n = 1,756):** adds operational variables (visit type, department, facility, workload index, admission, length of stay, referral, labs, doctor experience, demographics).
#
# Regression metric: cross-validated R² and MAE against a predict-the-mean baseline; classification of "Satisfied" (rating ≥ 4) uses ROC-AUC.

# %%
S1 = FIN.dropna(subset=["Rating"]).copy(); S1["ClaimRatio"] = S1.ClaimRatio.fillna(0); S1["ClaimStatus"] = S1.ClaimStatus.fillna("No claim"); S1["ClaimProvider"] = S1.ClaimProvider.fillna("No claim")
s1_num = ["BillAmount", "PaidRatio", "ClaimRatio", "HasClaim", "SeverityScore"]; s1_cat = ["Diagnosis", "Severity", "BillStatus", "PayMethod", "ClaimProvider", "ClaimStatus"]
S2 = Ew[Ew.Rating.notna()].copy()
s2_num = ["Age", "BMI", "RiskScore", "DoctorYearsExp", "PatientVisits", "Month_", "BillAmount", "PaidRatio", "DeptWorkloadIndex", "Admitted", "LabDone", "Referred", "SeverityScore"]
s2_cat = ["Gender", "ChronicCondition", "VisitType", "DepartmentName", "FacilityID", "Diagnosis", "Severity", "BillStatus", "PayMethod", "ClaimProvider"]
S2["ClaimProvider"] = S2.ClaimProvider.fillna("No claim"); S2["LengthOfStayDays"] = S2.LengthOfStayDays.fillna(0)
REGS = {"Baseline (mean)": lambda: DummyRegressor(), "Ridge": lambda: Ridge(alpha=50),
        "Random forest": lambda: RandomForestRegressor(200, min_samples_leaf=40, max_features=.4, n_jobs=-1, random_state=SEED),
        "Gradient boosting": lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=.04, max_iter=100, l2_regularization=10, random_state=SEED)}
cvR = RepeatedKFold(n_splits=5, n_repeats=2, random_state=SEED); sat_rows = []
for scope, df_, num_, cat_ in [("Scope 1 – all feedback (n=21,064)", S1, s1_num, s1_cat), ("Scope 2 – cohort + operational (n=1,756)", S2, s2_num, s2_cat)]:
    X_ = df_[num_ + cat_]; y_ = df_.Rating.values
    for mname, mk in REGS.items():
        cvr = cross_validate(make_pipe(num_, cat_, mk()), X_, y_, cv=cvR, scoring={"r2": "r2", "mae": "neg_mean_absolute_error"})
        sat_rows.append(dict(Scope=scope, Model=mname, R2=cvr["test_r2"].mean(), R2_sd=cvr["test_r2"].std(), MAE=-cvr["test_mae"].mean()))
    yb = (y_ >= 4).astype(int); aucs = {}
    for mname, mk in {"Logistic regression": lambda: LogisticRegression(C=.05, max_iter=2000), "Gradient boosting": CLFS["Gradient boosting"]}.items():
        a_ = cross_val_score(make_pipe(num_, cat_, mk()), X_, yb, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="roc_auc")
        sat_rows.append(dict(Scope=scope, Model=f"{mname} → P(rating ≥ 4)", R2=np.nan, R2_sd=np.nan, MAE=np.nan, AUC=a_.mean(), AUC_sd=a_.std()))
SAT = pd.DataFrame(sat_rows); R["satisfaction_models"] = SAT.round(4).to_dict("records")
pd.options.display.float_format = "{:,.4f}".format
SAT

# %%
# permutation test + importance for the scope-2 model; share-satisfied baseline
Xs, ys = S2[s2_num + s2_cat], S2.Rating.values
obs_r2 = SAT[(SAT.Scope.str.startswith("Scope 2")) & (SAT.Model == "Ridge")].R2.iloc[0]
cvs = KFold = None
from sklearn.model_selection import KFold
null_r2 = np.array([cross_val_score(make_pipe(s2_num, s2_cat, Ridge(alpha=50)), Xs, rng.permutation(ys), cv=KFold(5, shuffle=True, random_state=SEED), scoring="r2").mean() for _ in range(100)])
perm_p_sat = (1 + (null_r2 >= obs_r2).sum()) / (1 + len(null_r2))
rf_s = make_pipe(s2_num, s2_cat, REGS["Random forest"]()); Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=.25, random_state=SEED); rf_s.fit(Xtr, ytr)
pis = permutation_importance(rf_s, Xte, yte, scoring="r2", n_repeats=15, random_state=SEED)
IMP_S = pd.DataFrame({"Feature": Xte.columns, "Importance": pis.importances_mean, "sd": pis.importances_std}).sort_values("Importance", ascending=False).head(10)
best_s = SAT.dropna(subset=["R2"])[~SAT.Model.str.startswith("Baseline")].sort_values("R2", ascending=False).iloc[0]
R["sat_perm_p"], R["sat_best"] = float(perm_p_sat), best_s.to_dict(); R["sat_holdout_r2"] = float(r2_score(yte, rf_s.predict(Xte)))
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
reg = SAT.dropna(subset=["R2"]); sc_ = list(reg.Scope.unique())
for j, sc in enumerate(sc_):
    d_ = reg[reg.Scope == sc]; ax[0].bar(np.arange(len(d_)) + (j - .5) * .38, d_.R2, width=.36, yerr=d_.R2_sd, color=C[j], label=sc.split(" (")[0], ecolor=MUTED)
ax[0].axhline(0, color=INK2, lw=1); ax[0].set_xticks(range(4)); ax[0].set_xticklabels([m.replace(" ", "\n", 1) for m in reg.Model.unique()], fontsize=8)
ax[0].set(title="Cross-validated R² for predicting rating", ylabel="R² (0 = as good as predicting the mean)"); ax[0].legend(fontsize=8)
ax[1].hist(null_r2, bins=16, color=MUTED, edgecolor=SURF); ax[1].axvline(obs_r2, color=C[7], lw=2); ax[1].set(title="Scope 2 ridge R² vs label-shuffled null", xlabel="R² with shuffled ratings", ylabel="Count")
ax[1].text(obs_r2, ax[1].get_ylim()[1] * .9, f" observed {obs_r2:.3f}\n {fmt_p(perm_p_sat)}", color=C[7], fontsize=8.5, va="top")
ax[2].barh(IMP_S.Feature[::-1], IMP_S.Importance[::-1], xerr=IMP_S["sd"][::-1], color=C[0], height=.6, ecolor=MUTED); ax[2].axvline(0, color=INK2, lw=1)
ax[2].set(title="Permutation importance (random forest, hold-out)", xlabel="Mean R² decrease")
fig.tight_layout()
auc_rows = SAT.dropna(subset=["AUC"])
save_fig(fig, "f32_satisfaction_model",
         f"No model predicts satisfaction: the best cross-validated R² of any real model is {best_s.R2:.3f} ({best_s.Model}, {best_s.Scope.split(' (')[0]}) — zero or negative means no better than always guessing the average rating (≈{ys.mean():.2f}). "
         f"Classifying 'satisfied' (rating ≥ 4) gives AUC {auc_rows.AUC.min():.2f}–{auc_rows.AUC.max():.2f}. The label-shuffle test for the cohort model is {fmt_p(perm_p_sat)}. "
         "Clinical, billing, operational and demographic variables carry no information about the rating, so improving satisfaction needs new data — waiting time, staff attitude, cleanliness, drug availability — not a better model.")

save(globals(), __file__)
print("Done:", os.path.basename(__file__))
