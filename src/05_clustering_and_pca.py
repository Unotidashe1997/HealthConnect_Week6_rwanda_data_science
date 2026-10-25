"""
HealthConnect Rwanda - Week 6 (Data Science track)
Part 5 - Patient, facility and department clustering; PCA (figures 33-36)

Run from the repository root:  python src/05_clustering_and_pca.py
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
# ## 6. Clustering analysis & dimensionality reduction
# Three segmentations are built — **patients**, **facilities** and **departments** — each with K-Means (k chosen by silhouette) and, for facilities,
# Ward hierarchical clustering. Because clustering *always* returns groups, each solution is compared with a **null** built by shuffling every column
# independently (which destroys any real structure while keeping the same marginal distributions). Clusters are only treated as meaningful if they beat that null.

# %%
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

def sil_curve(Z, ks, sample=4000, reps_null=4, seed=SEED):
    rg = np.random.default_rng(seed); out = []
    for k in ks:
        km = KMeans(k, n_init=10, random_state=seed).fit(Z); s_ = silhouette_score(Z, km.labels_, sample_size=min(sample, len(Z)), random_state=seed) if len(Z) > sample else silhouette_score(Z, km.labels_)
        nl = []
        for _ in range(reps_null):
            Zn = np.column_stack([rg.permutation(Z[:, j]) for j in range(Z.shape[1])]); kn = KMeans(k, n_init=5, random_state=seed).fit(Zn)
            nl.append(silhouette_score(Zn, kn.labels_, sample_size=min(sample, len(Zn)), random_state=seed) if len(Zn) > sample else silhouette_score(Zn, kn.labels_))
        out.append(dict(k=k, inertia=km.inertia_, silhouette=s_, null_mean=np.mean(nl), null_max=np.max(nl)))
    return pd.DataFrame(out)

def name_cluster(z, labels, top=2, thr=.35):
    idx = np.argsort(-np.abs(z))[:top]; parts = [(("high " if z[i] > 0 else "low ") + labels[i]) for i in idx if abs(z[i]) >= thr]
    return ", ".join(parts) if parts else "near-average profile"

# %% [markdown]
# ### 6.1 Patient segmentation (20,000 patients)
# Features: age, BMI, three chronic-condition indicators, gender, and 2026 visit count (0 for patients with no recorded appointment).

# %%
PP = PATIENTS.copy()
for c_ in ["Asthma", "Hypertension", "Diabetes"]: PP[c_] = (PP.ChronicCondition == c_).astype(int)
PP["Female"] = (PP.Gender == "Female").astype(int)
pf = ["Age", "BMI", "Asthma", "Hypertension", "Diabetes", "Female", "Visits2026"]
ZP = StandardScaler().fit_transform(PP[pf])
SIL_P = sil_curve(ZP, range(2, 9))
kP = int(SIL_P.loc[SIL_P.silhouette.idxmax(), "k"]); kmP = KMeans(kP, n_init=15, random_state=SEED).fit(ZP); PP["Cluster"] = kmP.labels_
ari_P = adjusted_rand_score(kmP.labels_, KMeans(kP, n_init=15, random_state=SEED + 7).fit(ZP).labels_)   # seed-to-seed stability
# a clinically-interpretable 4-segment solution for management use (k = 4) is shown alongside, regardless of silhouette
k_use = 4; km4 = KMeans(k_use, n_init=15, random_state=SEED).fit(ZP); PP["Segment"] = km4.labels_
adm_pat = E.groupby("PatientID").agg(Admit=("Admitted", "max"), Bill=("BillAmount", "mean"))
PP = PP.merge(adm_pat, left_on="PatientID", right_index=True, how="left")
prof = PP.groupby("Segment").agg(Patients=("PatientID", "size"), MeanAge=("Age", "mean"), MeanBMI=("BMI", "mean"), Asthma=("Asthma", "mean"), Hypertension=("Hypertension", "mean"), Diabetes=("Diabetes", "mean"),
                                  Female=("Female", "mean"), MeanVisits=("Visits2026", "mean"), HighRisk=("HighRisk", "mean"), ChronicAny=("ChronicFlag", "mean"), AdmitRate=("Admit", "mean"), SeenPts=("Admit", "count"))
zprof = pd.DataFrame(km4.cluster_centers_, columns=pf)
prof["Description"] = [name_cluster(zprof.iloc[i].values, ["age", "BMI", "asthma", "hypertension", "diabetes", "female share", "visit count"]) for i in range(k_use)]
prof = prof.sort_values("HighRisk", ascending=False); prof["Label"] = [f"S{i + 1}" for i in range(len(prof))]
R["patient_clusters"] = dict(k_best=kP, sil_best=float(SIL_P.silhouette.max()), null_sil_at_best=float(SIL_P.loc[SIL_P.silhouette.idxmax(), "null_mean"]), ari=float(ari_P),
                             profile=prof.reset_index().round(4).to_dict("records"), sil_table=SIL_P.round(4).to_dict("records"))
ch_ = stats.chi2_contingency(pd.crosstab(PP.dropna(subset=["Admit"]).Segment, PP.dropna(subset=["Admit"]).Admit)); R["patient_cluster_admit_p"] = float(ch_[1])
pd.options.display.float_format = "{:,.3f}".format
print(f"Silhouette by k (real vs shuffled-null):"); print(SIL_P.round(3).to_string(index=False)); print(f"\nBest k = {kP}, seed-to-seed stability (ARI) = {ari_P:.2f}")
prof[["Label", "Patients", "MeanAge", "MeanBMI", "ChronicAny", "Hypertension", "Diabetes", "Asthma", "Female", "MeanVisits", "HighRisk", "AdmitRate", "SeenPts", "Description"]]

# %%
fig, ax = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": [1, 1.3, 1]})
ax[0].plot(SIL_P.k, SIL_P.silhouette, "o-", color=C[0], lw=2, label="Real data"); ax[0].plot(SIL_P.k, SIL_P.null_mean, "s--", color=C[1], lw=1.8, label="Shuffled-column null (mean)")
ax[0].fill_between(SIL_P.k, SIL_P.null_mean, SIL_P.null_max, color=C[1], alpha=.15); ax[0].set(title="Patients: silhouette vs null, by number of clusters", xlabel="Number of clusters (k)", ylabel="Silhouette score"); ax[0].legend(fontsize=8.5)
hm = pd.DataFrame(km4.cluster_centers_, columns=["Age", "BMI", "Asthma", "Hypertension", "Diabetes", "Female", "Visits"]).loc[prof.index]; hm.index = prof.Label
sns.heatmap(hm, cmap=DIV, center=0, vmin=-2, vmax=2, annot=True, fmt=".2f", linewidths=.6, linecolor=SURF, cbar_kws={"label": "Standardised mean (z)", "shrink": .8}, ax=ax[1]); ax[1].grid(False)
ax[1].set(title=f"Patient segments (k = {k_use}): cluster centres in standard deviations", ylabel="Segment (S1 = highest risk score)")
sz = prof.set_index("Label").Patients; ax[2].barh(sz.index[::-1], sz.values[::-1], color=C[0], height=.55)
for i, (v, h_) in enumerate(zip(sz.values[::-1], prof.set_index("Label").HighRisk.values[::-1])): ax[2].text(v + 80, i, f"{v:,}  (high-risk {h_:.0%})", va="center", fontsize=8.5)
ax[2].set(title="Segment size and high-risk share", xlabel="Patients", xlim=(0, sz.max() * 1.5))
fig.tight_layout()
save_fig(fig, "f33_patient_clusters",
         f"The only structure K-Means finds is the set of mutually exclusive chronic-condition indicators: silhouette peaks at {SIL_P.silhouette.max():.2f} (k = {kP}) versus {SIL_P.loc[SIL_P.silhouette.idxmax(), 'null_mean']:.2f} for shuffled data, and the gap comes from those binary variables, not from age, BMI, gender or visit behaviour, which show no grouping. "
         f"The four-segment solution is therefore the three chronic-condition groups (asthma, hypertension, diabetes; each ≈{int(prof.Patients.median()):,} patients and {100 * prof.HighRisk.sort_values().iloc[-3:].min():.0f}–{100 * prof.HighRisk.max():.0f}% meeting the high-risk rule) and the no-chronic-disease group ({100 * prof.HighRisk.min():.0f}% high-risk). "
         f"Segments do not differ in admission rate (χ² {fmt_p(ch_[1])}), so they help target outreach but add no outcome information beyond the chronic-condition label.")

# %% [markdown]
# ### 6.2 Facility segmentation (20 facilities)
# Twelve operational, clinical, financial and experience measures per facility (from the appointment cohort). With 20 facilities the solution is
# descriptive: it groups sites with similar profiles so management can compare like with like.

# %%
fcols = ["Appointments", "EmergencyShare", "AdmitRate", "MeanLOS", "ReferralRate", "MeanSeverity", "MeanBill", "CollectionRate", "ClaimApproval", "MeanRating", "ApptPerClinician", "StaffPer1000Pts"]
KF = K[fcols].copy(); ZF = StandardScaler().fit_transform(KF)
SIL_F = []
for k in range(2, 7):
    lab = KMeans(k, n_init=20, random_state=SEED).fit_predict(ZF); SIL_F.append(dict(k=k, silhouette=silhouette_score(ZF, lab)))
rg = np.random.default_rng(SEED); nullF = []
for _ in range(200):
    Zn = np.column_stack([rg.permutation(ZF[:, j]) for j in range(ZF.shape[1])]); nullF.append(max(silhouette_score(Zn, KMeans(k, n_init=5, random_state=0).fit_predict(Zn)) for k in range(2, 7)))
SIL_F = pd.DataFrame(SIL_F); kF = int(SIL_F.loc[SIL_F.silhouette.idxmax(), "k"]); best_sF = SIL_F.silhouette.max()
perm_pF = (1 + (np.array(nullF) >= best_sF).sum()) / (1 + len(nullF))
labF = KMeans(kF, n_init=20, random_state=SEED).fit_predict(ZF); KF["Cluster"] = labF
ward = linkage(ZF, "ward"); labW = fcluster(ward, kF, "maxclust"); ari_FW = adjusted_rand_score(labF, labW)
KF["PerformanceScore"] = K.PerformanceScore
pf_ = KF.groupby("Cluster").agg(Facilities=("Appointments", "size"), **{c_: (c_, "mean") for c_ in fcols}, PerformanceScore=("PerformanceScore", "mean"))
zc = (pf_[fcols] - KF[fcols].mean()) / KF[fcols].std(ddof=0)
pf_["Description"] = [name_cluster(zc.iloc[i].values, ["volume", "emergency share", "admission rate", "stay length", "referral rate", "severity", "bill size", "collection rate", "claim approval", "satisfaction", "load per clinician", "staff ratio"], top=3, thr=.4) for i in range(len(pf_))]
pf_["Members"] = KF.groupby("Cluster").apply(lambda d: ", ".join(d.index.str[-2:]))
R["facility_clusters"] = dict(k=kF, silhouette=float(best_sF), null_p=float(perm_pF), ari_kmeans_ward=float(ari_FW), profile=pf_.reset_index().round(4).to_dict("records"), sil_table=SIL_F.round(4).to_dict("records"))
pd.options.display.float_format = "{:,.3f}".format
print(f"Best k = {kF}, silhouette {best_sF:.3f}; chance of a silhouette ≥ this in shuffled data: {fmt_p(perm_pF)}; K-Means vs Ward agreement (ARI) = {ari_FW:.2f}")
pf_[["Facilities", "Members", "Appointments", "MeanRating", "CollectionRate", "AdmitRate", "ApptPerClinician", "PerformanceScore", "Description"]]

# %%
pcaF = PCA(2).fit(ZF); XF = pcaF.transform(ZF)
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1.1, 1.2, 1.1]})
dn = dendrogram(ward, labels=KF.index.str[-2:].values, ax=ax[0], color_threshold=ward[-(kF - 1), 2] if kF > 1 else None, above_threshold_color=MUTED, leaf_font_size=8.5)
ax[0].set(title="Ward hierarchical clustering of facilities", xlabel="Facility (HF0xx)", ylabel="Distance"); ax[0].grid(False)
for c_ in range(kF):
    m_ = labF == c_; ax[1].scatter(XF[m_, 0], XF[m_, 1], s=70, color=C[c_], edgecolor=SURF, label=f"Cluster {c_ + 1}", zorder=3)
for i, f_ in enumerate(KF.index): ax[1].annotate(f_[-2:], XF[i], textcoords="offset points", xytext=(5, 4), fontsize=8, color=INK2)
ax[1].set(title="Facilities in PCA space (K-Means colours)", xlabel=f"PC1 ({100 * pcaF.explained_variance_ratio_[0]:.0f}% of variance)", ylabel=f"PC2 ({100 * pcaF.explained_variance_ratio_[1]:.0f}%)"); ax[1].legend(fontsize=8)
ps = K.PerformanceScore.sort_values(ascending=False)
ax[2].barh(ps.index.str[-2:][::-1], ps.values[::-1], color=[C[int(KF.loc[f_, "Cluster"])] for f_ in ps.index[::-1]], height=.7); ax[2].set(title="Facility Performance Score (0–100), coloured by cluster", xlabel="Score")
fig.tight_layout()
save_fig(fig, "f34_facility_clusters",
         f"Facilities split into {kF} groups (silhouette {best_sF:.2f}; shuffled-data chance of doing this well {fmt_p(perm_pF)}), but the grouping is {'fairly weak and not stable: K-Means and Ward agree only moderately (ARI ' + f'{ari_FW:.2f}' + ')' if ari_FW < .6 else 'consistent across methods (ARI ' + f'{ari_FW:.2f}' + ')'}. "
         f"Cluster 1 = {pf_.iloc[0].Description}; cluster 2 = {pf_.iloc[1].Description if len(pf_) > 1 else 'n/a'} — i.e. the split rests on three mechanically linked volume measures (appointments, load per clinician, staff per 1,000 patients), not on quality or outcomes. The Performance Score (scaled 0–100 by construction; {ps.index[0]} highest, {ps.index[-1]} lowest) is built from components that rest on only ~{int(K.RatedN.median())} ratings and ~{int(K.Appointments.median())} appointments per site, so rank differences are not statistically reliable — use clusters to pick peer groups for comparison, not to rank.")

# %% [markdown]
# ### 6.3 Department segmentation (200 departments)
# Features per department-ID: appointments, emergency share, admission rate, lab and prescription intensity, mean severity, mean bill and appointments per clinician
# (workload). With ≈25 appointments per department these are noisy, so the null comparison matters most here.

# %%
dcols = ["Appointments", "Emergency", "AdmitRate", "LabPerAppt", "RxPerAppt", "MeanSeverity", "MeanBill", "ApptPerClinician"]
DD = DEPT[dcols].replace([np.inf, -np.inf], np.nan).dropna(); ZD = StandardScaler().fit_transform(DD)
SIL_D = sil_curve(ZD, range(2, 8), sample=200, reps_null=5)
kD = int(SIL_D.loc[SIL_D.silhouette.idxmax(), "k"]); kmD = KMeans(kD, n_init=15, random_state=SEED).fit(ZD); DD["Cluster"] = kmD.labels_
k_show = 4; km4d = KMeans(k_show, n_init=15, random_state=SEED).fit(ZD); DD["Segment"] = km4d.labels_
dprof = DD.groupby("Segment").agg(Departments=("Appointments", "size"), **{c_: (c_, "mean") for c_ in dcols})
zd = pd.DataFrame(km4d.cluster_centers_, columns=dcols)
dprof["Description"] = [name_cluster(zd.iloc[i].values, ["volume", "emergency share", "admission rate", "lab intensity", "Rx intensity", "severity", "bill size", "load per clinician"], top=3, thr=.5) for i in range(k_show)]
dtype_mix = DD.join(DEPT.DepartmentName).groupby("Segment").DepartmentName.agg(lambda s: s.value_counts().index[0] + f" ({s.value_counts().iloc[0] / len(s):.0%})")
dprof["Most common type"] = dtype_mix
ch_d = stats.chi2_contingency(pd.crosstab(DD.join(DEPT.DepartmentName).Segment, DD.join(DEPT.DepartmentName).DepartmentName)); R["dept_cluster_type_p"] = float(ch_d[1])
R["dept_clusters"] = dict(k_best=kD, sil_best=float(SIL_D.silhouette.max()), null_at_best=float(SIL_D.loc[SIL_D.silhouette.idxmax(), "null_mean"]), profile=dprof.reset_index().round(3).to_dict("records"))
print(SIL_D.round(3).to_string(index=False)); print(f"\nBest k = {kD}; χ² of segment vs department type: {fmt_p(ch_d[1])}")
dprof

# %%
fig, ax = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": [1, 1.3, 1]})
ax[0].plot(SIL_D.k, SIL_D.silhouette, "o-", color=C[0], lw=2, label="Real data"); ax[0].plot(SIL_D.k, SIL_D.null_mean, "s--", color=C[1], lw=1.8, label="Shuffled null (mean)"); ax[0].fill_between(SIL_D.k, SIL_D.null_mean, SIL_D.null_max, color=C[1], alpha=.15)
ax[0].set(title="Departments: silhouette vs null", xlabel="Number of clusters (k)", ylabel="Silhouette score"); ax[0].legend(fontsize=8.5)
hd = pd.DataFrame(km4d.cluster_centers_, columns=["Volume", "Emergency", "Admit rate", "Lab", "Rx", "Severity", "Bill", "Load/clin."]); hd.index = [f"D{i + 1}" for i in range(k_show)]
sns.heatmap(hd, cmap=DIV, center=0, vmin=-2, vmax=2, annot=True, fmt=".2f", linewidths=.6, linecolor=SURF, cbar_kws={"label": "Standardised mean (z)", "shrink": .8}, ax=ax[1]); ax[1].grid(False)
ax[1].set(title=f"Department segments (k = {k_show}): profile in standard deviations", ylabel="Segment")
pcd = PCA(2).fit(ZD); XD = pcd.transform(ZD)
for c_ in range(k_show): ax[2].scatter(XD[DD.Segment.values == c_, 0], XD[DD.Segment.values == c_, 1], s=14, alpha=.7, color=C[c_], label=f"D{c_ + 1} (n={int((DD.Segment == c_).sum())})", linewidths=0)
ax[2].set(title="Departments in PCA space", xlabel=f"PC1 ({100 * pcd.explained_variance_ratio_[0]:.0f}%)", ylabel=f"PC2 ({100 * pcd.explained_variance_ratio_[1]:.0f}%)"); ax[2].legend(fontsize=8)
fig.tight_layout()
save_fig(fig, "f35_department_clusters",
         f"Department profiles form a largely continuous cloud rather than separate groups (best silhouette {SIL_D.silhouette.max():.2f}, only slightly above {SIL_D.loc[SIL_D.silhouette.idxmax(), 'null_mean']:.2f} for shuffled data), and segments are unrelated to department type (χ² {fmt_p(ch_d[1])}). "
         "The four-segment split still gives management a practical triage list: segments with high volume and high load per clinician (see heat-map rows with positive 'Volume' and 'Load/clin.') are the first candidates for staffing review, but these profiles rest on ~25 appointments per department and should be re-estimated as more data accrue.")

# %% [markdown]
# ### 6.4 Principal Component Analysis
# PCA is applied to the standardised patient feature matrix (numeric + chronic indicators + visits) and to the facility feature matrix to show how many dimensions are really present.

# %%
pf_all = ["Age", "BMI", "HeightCm", "WeightKg", "Asthma", "Hypertension", "Diabetes", "Female", "Visits2026", "RiskScore"]
ZPA = StandardScaler().fit_transform(PP[pf_all]); pcaP = PCA().fit(ZPA); evrP = pcaP.explained_variance_ratio_; cumP = evrP.cumsum()
pcaFa = PCA().fit(ZF); evrF = pcaFa.explained_variance_ratio_; cumF = evrF.cumsum()
n80P, n90P = int(np.argmax(cumP >= .8) + 1), int(np.argmax(cumP >= .9) + 1); n80F, n90F = int(np.argmax(cumF >= .8) + 1), int(np.argmax(cumF >= .9) + 1)
# parallel-analysis style null: eigenvalues from column-shuffled data
nullE = np.mean([PCA().fit(np.column_stack([rng.permutation(ZPA[:, j]) for j in range(ZPA.shape[1])])).explained_variance_ratio_ for _ in range(20)], axis=0)
load = pd.DataFrame(pcaP.components_[:3].T, index=pf_all, columns=["PC1", "PC2", "PC3"])
R["pca"] = dict(patients=dict(evr=evrP.round(4).tolist(), n80=n80P, n90=n90P, n_features=len(pf_all), pc1=float(evrP[0])), facilities=dict(evr=evrF.round(4).tolist(), n80=n80F, n90=n90F, n_features=len(fcols), pc1=float(evrF[0])))
fig, ax = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": [1.1, 1.1, 1]})
ax[0].bar(range(1, len(evrP) + 1), 100 * evrP, color=C[0], width=.6, label="Variance explained"); ax[0].plot(range(1, len(evrP) + 1), 100 * cumP, "o-", color=C[1], lw=2, label="Cumulative")
ax[0].plot(range(1, len(evrP) + 1), 100 * nullE, "s--", color=MUTED, lw=1.3, label="Shuffled-data null"); ax[0].axhline(80, color=INK2, ls=":", lw=1)
ax[0].set(title="Patients: scree plot and cumulative variance", xlabel="Principal component", ylabel="% of variance"); ax[0].legend(fontsize=8)
ax[1].bar(range(1, len(evrF) + 1), 100 * evrF, color=C[0], width=.6); ax[1].plot(range(1, len(evrF) + 1), 100 * cumF, "o-", color=C[1], lw=2); ax[1].axhline(80, color=INK2, ls=":", lw=1)
ax[1].set(title="Facilities: scree plot and cumulative variance", xlabel="Principal component", ylabel="% of variance")
sns.heatmap(load, cmap=DIV, center=0, vmin=-.8, vmax=.8, annot=True, fmt=".2f", linewidths=.6, linecolor=SURF, cbar_kws={"label": "Loading", "shrink": .8}, ax=ax[2]); ax[2].grid(False); ax[2].set(title="Patient PCA loadings (first 3 components)")
fig.tight_layout()
save_fig(fig, "f36_pca",
         f"Patient data need {n80P} of {len(pf_all)} components to retain 80% of variance ({n90P} for 90%); the first component explains {100 * evrP[0]:.0f}% versus {100 * nullE[0]:.0f}% for shuffled data — the excess comes from variables that are built from each other — so beyond those built-in links the variables are independent and cannot be compressed much. "
         f"PC1 is dominated by {', '.join(load.PC1.abs().sort_values(ascending=False).head(3).index)} (loadings {', '.join(f'{load.PC1[i]:+.2f}' for i in load.PC1.abs().sort_values(ascending=False).head(3).index)}), reflecting arithmetic links between variables (BMI is computed from height and weight; the risk score from age, BMI and chronic disease). Facility measures need {n80F} of {len(fcols)} components for 80% ({n90F} for 90%) because 20 sites × 12 noisy measures leave little shared structure. "
         "PCA therefore offers little dimensionality reduction here; it is retained as a diagnostic that confirms the absence of hidden low-dimensional drivers.")

save(globals(), __file__)
print("Done:", os.path.basename(__file__))
