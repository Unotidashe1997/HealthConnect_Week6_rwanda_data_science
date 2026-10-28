"""
HealthConnect Rwanda - Week 7 (Data Science track)
Builds the Week 7 documents from outputs/results.json, outputs/tables and outputs/figures:
  reports/HC_Week7_Model_Testing_and_Refinement_Report.docx   (main track deliverable)
  reports/HC_Week7_Project_Summary.docx                       (concise summary required by the brief)
  reports/HC_Week7_Testing_Validation_Record.xlsx             (testing record, issue log, cross-track record)
Run from the week7 folder after src/week7_model_testing.py:  python src/build_week7_docs.py
"""
import json, os
import pandas as pd
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.makedirs("outputs/reports", exist_ok=True)
J = json.load(open("outputs/results.json")); R, FIGS = J["R"], J["FIGS"]
W6 = json.load(open("../outputs/results.json"))["R"]
BLUE = RGBColor(0x2a, 0x78, 0xd6); GREY = RGBColor(0x55, 0x55, 0x55)
pc = lambda x, d=1: f"{100 * x:.{d}f}%"

def new_doc(title, sub):
    d = Document(); s = d.sections[0]; s.left_margin = s.right_margin = Cm(2); s.top_margin = s.bottom_margin = Cm(1.9)
    d.styles["Normal"].font.name = "Calibri"; d.styles["Normal"].font.size = Pt(10.5)
    for h, sz in (("Heading 1", 16), ("Heading 2", 12.5), ("Heading 3", 11)):
        d.styles[h].font.name = "Calibri"; d.styles[h].font.size = Pt(sz); d.styles[h].font.color.rgb = BLUE
    p = d.add_paragraph(); r = p.add_run(title); r.bold = True; r.font.size = Pt(22); r.font.color.rgb = BLUE
    p = d.add_paragraph(); r = p.add_run(sub); r.font.size = Pt(11); r.font.color.rgb = GREY
    p = d.add_paragraph(); r = p.add_run("HealthConnect Clinic Experience Lab · AnalystLab Africa Experience Lab, Week 7 · Data Science track · Testing, Refinement & End-to-End Validation"); r.italic = True; r.font.size = Pt(9)
    return d
def shade(cell, col):
    tcPr = cell._tc.get_or_add_tcPr(); sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), col); tcPr.append(sh)
def para(d, t, bold=False, italic=False, size=None, after=6):
    p = d.add_paragraph(); r = p.add_run(t); r.bold = bold; r.italic = italic
    if size: r.font.size = Pt(size)
    p.paragraph_format.space_after = Pt(after); return p
def bullets(d, items):
    for t in items:
        p = d.add_paragraph(style="List Bullet")
        if isinstance(t, tuple): r = p.add_run(t[0]); r.bold = True; p.add_run(t[1])
        else: p.add_run(t)
        p.paragraph_format.space_after = Pt(2)
def box(d, text, fill="E8F1FC", bold_first=None):
    t = d.add_table(rows=1, cols=1); t.style = "Table Grid"; c = t.rows[0].cells[0]; shade(c, fill); c.text = ""
    p = c.paragraphs[0]
    if bold_first: r = p.add_run(bold_first + " "); r.bold = True
    p.add_run(text); d.add_paragraph().paragraph_format.space_after = Pt(2)
def table(d, df, widths=None, size=8.5, colorcol=None):
    t = d.add_table(rows=1, cols=len(df.columns)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, c in enumerate(df.columns):
        cell = t.rows[0].cells[i]; cell.text = ""; r = cell.paragraphs[0].add_run(str(c)); r.bold = True; r.font.size = Pt(size); r.font.color.rgb = RGBColor(255, 255, 255); shade(cell, "2A78D6")
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(row):
            if isinstance(v, float): v = "" if pd.isna(v) else (f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3f}" if abs(v) < 10 else f"{v:,.1f}")
            cells[i].text = ""; r = cells[i].paragraphs[0].add_run("" if v is None else str(v)); r.font.size = Pt(size)
            if colorcol and df.columns[i] in colorcol:
                sv = str(v)
                if sv.startswith("PASS"): shade(cells[i], "D5F0E4")
                elif sv.startswith("FAIL"): shade(cells[i], "FADBDB")
    if widths:
        t.autofit = False
        for row in t.rows:
            for i, w in enumerate(widths): row.cells[i].width = Cm(w)
    d.add_paragraph().paragraph_format.space_after = Pt(2); return t
def fig(d, key, width=16.5):
    name = [k for k in FIGS if k.startswith(key)][0]; d.add_heading(f"Figure {key.upper()}. {FIGS[name]['title']}", 3)
    d.add_picture(f"outputs/figures/{name}.png", width=Cm(width)); d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

# ------------------------------------------------------------------ numbers ------------------------------------------------------------------
S, CAP, CAL, HF, CR, IF = R["stability"], R["capacity_rule"], R["calibration"], R["holdout_final"], R["criteria"], R["interface"]
SEG, IMP, EP, CC, FV = R["segments"], R["importance"], R["error_profile"], R["conf_counts"], R["flag_vs_unflag"]
BR = R["best_refinement"]; TS = R["tests_summary"]; REF = HF["ref"]; CAND = HF["cand"]
miss = EP["Who is missed? (admitted: missed vs caught)"]; wrong = EP["Who is wrongly flagged? (not admitted: flagged vs not)"]
tests = pd.DataFrame(R["tests"]); VT = pd.DataFrame(R["refine_dev"]); M1 = pd.DataFrame(R["metrics_candidate"]); XT = pd.DataFrame(IF["table"])
OV = pd.DataFrame(R["overfit"]); PL = pd.DataFrame(R["planted"]); LEAK = pd.DataFrame(R["leak_audit"]); CRT = pd.DataFrame(CR["table"]).rename(columns={"index": "Criterion"})
n_fail_if = IF["total"] - IF["v1_pass"]

# ------------------------------------------------------------------ issue log & cross-track record ------------------------------------------------------------------
ISSUES = pd.DataFrame([
 ["I-01", "Candidate admission model has no predictive skill (hold-out AUC %.3f, CI includes 0.50; 0 of 6 acceptance criteria met)" % CAND["auc"], "Critical", "Testing T02–T15", "Open — cannot be fixed with this data", "Data Science", "Do not use for patient-level decisions; keep rule-based risk list; revisit after Week 8 data additions", "Re-tested after 7 refinements: no improvement (T12–T15)"],
 ["I-02", "Brief assumes a missed-appointment (no-show) model on HealthConnect_Appointment_Data.csv; project data contain no attendance field", "Critical (dependency)", "Week 6→7 transition", "Open — needs decision", "Project Management / Data owner", "Obtain the appointment/attendance extract or confirm the admission model as the agreed proxy; re-use this test harness on the new target", "n/a"],
 ["I-03", "Week 6 scoring interface (V1) accepted impossible values, missing fields and unreported extra columns; returned a bare array (%d of %d interface tests failed)" % (n_fail_if, IF["total"]), "High", "Interface tests X01–X18", "Resolved", "Data Science → ML Engineering", "Input/output contract, validation layer, documented output schema (V2)", f"V2 passes {IF['v2_pass']}/{IF['total']}"],
 ["I-04", "CSV 'None' (no chronic disease) is read as missing by default; V1 silently scored it as a category", "Medium", "X14, X15", "Resolved (guard in V2); recurring risk", "Data Science / ML Engineering", "V2 rejects missing ChronicCondition with a hint; read CSVs with keep_default_na=False", "X15 PASS in V2"],
 ["I-05", "PatientVisits counts the whole year, so it includes visits after the one being scored (mild look-ahead)", "Medium", "T09 / feature-timing audit", "Open", "Data Science / ML Engineering", "Recompute as 'visits before this appointment date' in the production pipeline", "Not re-tested (no signal either way)"],
 ["I-06", "Diagnosis, severity, labs, prescriptions and referral are only known after the visit (leakage if used at booking time)", "Medium", "Feature-timing audit", "Resolved", "Data Science", "Excluded from the deployed feature set; documented in model card", "n/a"],
 ["I-07", "Predicted probabilities are uninformative (calibration slope %.2f; Brier %.4f vs baseline %.4f)" % (CAL["slope"], CAL["brier"], CAL["brier_base"]), "High", "T05, T15", "Open — cannot be fixed with this data", "Data Science", "Do not display risk percentages; recalibration cannot add information", "Isotonic calibration: Brier %.4f (baseline %.4f)" % (HF["brier_calibrated"], HF["brier_base"])],
 ["I-08", "Facility, department and doctor keys are inconsistent in the source data (carried from Week 6); only 5,000 of 60,000 billed visits have an appointment record", "High", "Week 6 data-quality report", "Open", "Project Management / Data owner", "Re-extract appointments with enforced foreign keys before any facility-level modelling", "n/a"],
 ["I-09", "Only 12 months of data; the time-based test has %d appointments and %d admissions" % (S["time_n"], S["time_pos"]), "Medium", "T03", "Open", "Data Science", "Collect 24+ months; repeat the time-based test", "n/a"],
 ["I-10", "ML Engineering teammate has not yet run the hand-off tests or signed off the contract", "Medium", "Cross-track record", "Open", "Intern + ML Engineering teammate", "Share the hand-off package (model card, contract, scorer, tests) and record their result in the cross-track record", "Pending"],
 ["I-11", "No Week 5 baseline model exists in this project line, so the baseline is naive (prior probability, single-rule score)", "Low", "Model comparison", "Documented", "Data Science", "Naive baselines are the strictest available comparison", "n/a"],
], columns=["ID", "Issue", "Severity", "Found in", "Status", "Owner", "Action / mitigation", "Retest"])
XREC = pd.DataFrame([
 ["1. Track collaborated with", "ML Engineering (hand-off prepared by Data Science; teammate review pending). Supporting inputs: Data Analytics (Week 6 EDA/statistical findings reused in feature review); Project Management (issue log)."],
 ["2. Project dependency", "ML Engineering depends on a model with documented inputs, outputs and limits; Data Science depends on ML Engineering to confirm the pipeline preserves them."],
 ["3. Component/output being tested", "Scoring interface around the Week 6 candidate model: input contract, validation, output schema, persistence, speed."],
 ["4. Information/output received", "[To complete with ML Engineering teammate: pipeline/preprocessing details, their input format, their run results]"],
 ["5. Information/output provided", "Model card, input/output contract (JSON), validated scorer (admission_risk_scorer.py), saved model, 18-case interface test suite, accepted/rejected-input rules, caution that the model has no predictive skill."],
 ["6. Testing activity completed", f"18 interface test cases run on the Week 6 behaviour (V1) and the refined scorer (V2): V1 {IF['v1_pass']}/{IF['total']}, V2 {IF['v2_pass']}/{IF['total']}. [Teammate run: to be completed]"],
 ["7. Issue/finding identified", f"{n_fail_if} failures in V1: impossible/missing values scored silently, 'None'→NaN pitfall, unreported extra columns (possible identifiers), bare-array output, empty input crash, unclear error messages."],
 ["8. Refinement/action taken", "Input contract with ranges and closed lists; validation with field-named errors; unseen-category warnings; documented output schema with version and 'use caution' field; empty-input handling."],
 ["9. Retest result", f"V2 passes {IF['v2_pass']}/{IF['total']} (src/admission_risk_scorer.py; outputs/tables/t_scoring_interface_tests.csv)."],
 ["10. What changed as a result", "Scores can no longer be produced from invalid input without a clear error or warning; the pipeline owner now has an explicit contract and a caution flag."],
 ["11. Evidence", "Notebook section F; Figure W08; model_card.md; input_output_contract.json; models/admission_scorer_v2.joblib; [teammate confirmation/screenshot: to be added]"],
 ["12. How it improved HealthConnect", "Prevents silent bad scores and identifier leakage into the model pipeline, and makes the model's lack of skill visible to downstream users before Week 8 integration."]], columns=["Field", "Record"])
tests_full = tests.copy(); tests_full.insert(1, "Component_Tested", tests_full.ID.map(lambda i: "Week 6 candidate admission classifier" if i.startswith("T") else "Scoring interface (Week 6 model wrapper)"))
tests_full.insert(2, "Testing_Objective", tests_full.ID.map(lambda i: "Determine whether the model meets the pre-registered acceptance criteria" if i.startswith("T") else "Determine whether the interface handles valid, invalid and edge-case inputs per the contract"))
tests_full = tests_full.rename(columns={"Result": "Pass_Fail", "Issue": "Issue_Identified", "Action": "Action_Taken", "Retest": "Retest_Result"})
with pd.ExcelWriter("outputs/reports/HC_Week7_Testing_Validation_Record.xlsx") as xw:
    for nm, df_ in [("Testing record", tests_full), ("Integration & issue log", ISSUES), ("Cross-track record", XREC), ("Acceptance criteria", CRT), ("Interface tests", XT)]:
        df_.to_excel(xw, sheet_name=nm, index=False); ws = xw.sheets[nm]
        for i, c in enumerate(df_.columns): ws.set_column(i, i, float(min(70, max(12, df_[c].astype(str).str.len().quantile(.9) * .9))))
print("saved testing validation record")

# ------------------------------------------------------------------ MAIN REPORT ------------------------------------------------------------------
d = new_doc("Model Testing & Refinement Report", "Week 7 — testing, error analysis and refinement of the Week 6 candidate model")
d.add_heading("Verdict at a glance", 1)
box(d, f"The Week 6 candidate model does not meet any of the 6 pre-registered acceptance criteria and is not suitable for patient-level use. Seven evidence-based refinements did not change that (best: {BR['name']}, ΔAUC {BR['d']:+.3f}, p = {BR['p']:.2f}). "
       f"Testing the method itself showed it would have detected a real effect (planted signals recovered from a true AUC of about {R['min_detectable_auc']:.2f}), so the result reflects the data, not a weak test. "
       f"The most useful improvement was to the model's scoring interface: Week 6 behaviour passed {IF['v1_pass']} of {IF['total']} interface tests and the refined scorer passes {IF['v2_pass']} of {IF['total']}. "
       "The data also contain no attendance field, so a missed-appointment model — the project's real objective — cannot yet be built; this is the main item to resolve before Week 8.", bold_first="Conclusion.")
table(d, pd.DataFrame([["Hold-out AUC (untouched 1,000 appointments)", f"{CAND['auc']:.3f} (95% CI {CAND['lo']:.2f}–{CAND['hi']:.2f})", f"{REF['auc']:.3f} (95% CI {REF['lo']:.2f}–{REF['hi']:.2f})"],
                       ["Time-based test AUC (train Jan–Sep, test Oct–Dec)", f"{S['time_auc']:.3f} (CI {S['time_ci'][0]:.2f}–{S['time_ci'][1]:.2f})", f"{HF['time_ref']['auc']:.3f} (CI {HF['time_ref']['ci'][0]:.2f}–{HF['time_ref']['ci'][1]:.2f})"],
                       ["Share of 30 random splits with AUC > 0.5", pc(S["share_above_half"], 0), pc(HF["stability_ref"]["share"], 0)],
                       ["Calibration slope (ideal 1.0)", f"{CAL['slope']:.2f}", f"{HF['calib_ref']['slope']:.2f}"],
                       ["Acceptance criteria met", f"{CR['passed'][0]} of 6", f"{CR['passed'][1]} of 6"],
                       ["Interface tests passed", f"{IF['v1_pass']} of {IF['total']} (Week 6 behaviour)", f"{IF['v2_pass']} of {IF['total']} (refined scorer)"]],
                      columns=["Measure", "Week 6 candidate", f"Refined ({REF['name']})"]), [7, 5, 5], size=9)

d.add_heading("1. Week 6 → Week 7 transition", 1)
table(d, pd.DataFrame([
 ["1. Main Week 6 output", "Healthcare predictive-analytics package: integrated data (20,000 patients, 5,000 encounters, 60,000 bills), 36 charts, 15 hypothesis tests, forecasts, admission and satisfaction models, clustering, reports, 8-script GitHub pipeline."],
 ["2. Most important component", f"The candidate classification model: logistic regression predicting hospital admission from booking-time information (Week 6 CV AUC {W6['admission_best']['AUC']:.3f}, hold-out AUC {W6['admission_holdout']['auc']:.3f})."],
 ["3. Main issue / limitation / dependency", f"No predictive signal found in Week 6 (permutation p = {W6['admission_perm_p']:.2f}); model tested on one split and one threshold only; inconsistent keys across source tables; no attendance field."],
 ["4. Component requiring testing", "The candidate model and the way it will be called by the ML pipeline (scoring interface)."],
 ["5. Relevant tracks", "Data Science (owner), ML Engineering (interface/pipeline), Data Analytics (which findings should drive features), Project Management (issue log, readiness)."],
 ["6. What I intend to test", "Stability, metrics at deployable thresholds, calibration, errors (false positives/negatives), segments, overfitting, feature validity, whether the test method can detect a real signal, and the scoring interface."],
 ["7. What I expect to improve / validate", "Decide whether the model is fit for use, refine only on evidence, harden the interface for ML Engineering, and define what Week 8 needs."]], columns=["Item", "Answer"]), [4.5, 12.5], size=9)
box(d, "The Week 7 brief describes a missed-appointment (no-show) classifier on HealthConnect_Appointment_Data.csv. That file is not part of the project data used since Week 4 (the 27 HealthConnect hospital files), and those data have no attendance, no-show or cancellation field. "
       "As instructed (\"do not select a new dataset\"), this report tests the Week 6 candidate — hospital admission — and states in section 9 what that means for the missed-appointment objective.", fill="FFF4D6", bold_first="Important context.")

d.add_heading("2. Week 6 candidate model summary", 1)
table(d, pd.DataFrame([["Task", "Binary classification: will the appointment end in a hospital admission? (586 of 5,000 = 11.7%)"],
                       ["Features (18)", "Age, BMI, RiskScore, ChronicFlag, DoctorYearsExp, PatientVisits, Month; gender, marital status, education, occupation, province, blood group, chronic condition, visit type, department, facility, weekday"],
                       ["Algorithm", "Logistic regression (L2, C = 0.1), standardised numerics, one-hot categories (rare levels grouped)"],
                       ["Week 6 evidence", f"Repeated 5×3 CV AUC {W6['admission_best']['AUC']:.3f}; hold-out AUC {W6['admission_holdout']['auc']:.3f}; permutation p = {W6['admission_perm_p']:.2f}; richer feature sets and random forest / gradient boosting did no better"],
                       ["Baselines", "Prior-probability model (AUC 0.50) and a single-rule score (RiskScore only). No Week 5 model exists in this project line, so naive baselines — the strictest available — are used."],
                       ["Intended purpose", "Rank appointments by admission risk to target outreach and plan capacity"]], columns=["Item", "Detail"]), [3.5, 13.5], size=9)

d.add_heading("3. Testing approach", 1)
para(d, "Acceptance criteria were fixed before any testing, so the verdict cannot be tuned to the results. The model is suitable for patient-level use only if all six hold on untouched data.")
table(d, pd.DataFrame([["C1 Discrimination", "Hold-out ROC-AUC ≥ 0.65"], ["C2 Significance", "Lower 95% bootstrap bound of hold-out AUC > 0.50"], ["C3 Precision value", "PR-AUC ≥ 1.5 × prevalence"],
                       ["C4 Stability", "AUC > 0.5 in ≥ 90% of 30 random splits and in the time-based split (CI lower bound > 0.5)"], ["C5 Calibration", "Calibration slope between 0.8 and 1.2"], ["C6 Beats baseline", "Corrected paired CV gain over the naive baseline with a 95% interval excluding 0"]],
                      columns=["Criterion", "Rule"]), [4, 13], size=9)
bullets(d, [("Data splits. ", "Development set 4,000 appointments (selection and tuning, repeated 5×3 CV with identical folds for all models) and an untouched hold-out of 1,000 (same split as Week 6), opened once at the end. A separate time-based split (train Jan–Sep, test Oct–Dec) mimics deployment."),
            ("Test battery. ", f"{TS['n']} recorded tests: 15 model tests (T01–T15) and 18 scoring-interface tests (X01–X18). Each is recorded with expected result, actual result, pass/fail, issue, action and retest (Appendix A and the Excel record)."),
            ("Reading the results. ", "A FAIL on a model test (T-series) is a finding about the model, not a defect in the testing; failures are reported, not hidden. Where no issue exists the test is marked PASS (reproducibility, negative control, planted-signal detection).")])

d.add_heading("4. Test results and performance metrics", 1)
d.add_heading("4.1 Reproducibility, stability and time-based validation (T01–T03)", 2)
para(d, f"The Week 6 result reproduced exactly (CV AUC {R['repro']['cv_auc']:.3f}, hold-out AUC {R['repro']['ho_auc']:.3f}). It is not stable: across 30 random 80/20 splits the AUC averages {S['mean']:.3f} (sd {S['sd']:.3f}, range {S['lo']:.2f}–{S['hi']:.2f}) and only {pc(S['share_above_half'], 0)} of splits exceed 0.50 — about what chance alone would give. "
          f"On future appointments (Oct–Dec) the AUC is {S['time_auc']:.3f} (95% CI {S['time_ci'][0]:.2f}–{S['time_ci'][1]:.2f}).")
fig(d, "w01")
d.add_heading("4.2 Classification metrics at deployable thresholds (T04)", 2)
para(d, f"Thresholds were chosen on the development set only and applied once to the hold-out. A 0.5 threshold flags {CAP['flagged_05']} patients (predicted probabilities never reach it). The F1-optimal threshold ({CAP['t_f1']:.3f}) flags {CAP['flagged_f1']} of 1,000 patients — essentially everyone, equivalent to the 'flag everyone' baseline. "
          f"A realistic capacity rule (flag the top 20%) has precision {pc(CAP['precision'])} against a prevalence of 11.7% (lift {CAP['lift']:.2f}×, precision 95% CI {pc(CAP['p_lo'])}–{pc(CAP['p_hi'])}) while random selection gives {pc(CAP['random_precision'])}.")
m1 = M1[["Rule", "Flagged", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1", "Balanced_Acc", "MCC"]].copy()
table(d, m1, [4.2, 1.3, 1, 1, 1, 1.2, 1.5, 1.3, 1.2, 1.7, 1.3], size=7.5)
fig(d, "w02")
d.add_heading("4.3 Calibration (T05)", 2)
para(d, f"The calibration slope is {CAL['slope']:.2f} (95% CI {CAL['slope_ci'][0]:.2f}–{CAL['slope_ci'][1]:.2f}; ideal 1.0): observed admission rates barely change across risk deciles. The Brier score ({CAL['brier']:.4f}) is no better than predicting the average ({CAL['brier_base']:.4f}). Predicted risks therefore should not be shown to staff as percentages.")
fig(d, "w03")

d.add_heading("5. Error analysis", 1)
d.add_heading("5.1 False positives and false negatives (T06)", 2)
para(d, f"Using the capacity rule on out-of-fold predictions for all 5,000 appointments: {CC['TP']} true positives, {CC['FP']} false positives, {CC['FN']} false negatives and {CC['TN']:,} true negatives. {pc(CC['FN'] / (CC['TP'] + CC['FN']), 0)} of admissions are missed and {pc(CC['FP'] / (CC['TP'] + CC['FP']), 0)} of flags are false alarms. "
          f"Flagged patients are admitted at {pc(FV['flagged_rate'])} versus {pc(FV['unflagged_rate'])} for unflagged patients (chi-square p = {FV['p']:.2f}): the flags do not enrich for admissions.")
para(d, f"Patterns in the errors: missed-versus-caught admissions differ on {miss['holm']} of {miss['n']} variables and wrongly-flagged versus unflagged non-admitted patients on {wrong['holm']} of {wrong['n']} (Holm-adjusted), led by department, weekday and province. "
          "This is a selection effect rather than an insight: patients are flagged because of the values of the very variables the score uses, and those variables (department, weekday, province) had no demonstrated link to admission in Week 6. "
          "There is therefore no outcome-related error pattern that a feature fix could remove.")
d.add_heading("5.2 Performance across segments (T07)", 2)
para(d, f"AUC was assessed in {SEG['n']} segments (gender, age band, province, department, visit type, severity, chronic status; ≥ 30 admissions each). Values range {SEG['min_auc']:.2f}–{SEG['max_auc']:.2f}; {SEG['above']} intervals lie wholly above 0.50 and {SEG['below']} wholly below (about {SEG['expected_chance']:.1f} expected by chance); {SEG['holm_sig']} remain significant after Holm correction. The model is not reliable in any segment, and no segment is systematically worse.")
fig(d, "w04", 15)
d.add_heading("6. Robustness and validity tests", 1)
d.add_heading("6.1 Overfitting (T08)", 2)
ov = OV.copy(); ov.columns = ["Model", "Training AUC", "CV AUC", "Gap"]; table(d, ov, [8, 3, 3, 3], size=9)
para(d, "The candidate overfits mildly and, as data grows, its training AUC falls while its cross-validated AUC stays at about 0.5: it fits noise, not signal. Flexible models make this worse (the unconstrained forest memorises the training data and still scores 0.50 in cross-validation).")
fig(d, "w05")
d.add_heading("6.2 Do the features support the objective? (T09)", 2)
para(d, f"Permutation importance on five held-out folds: the largest mean importance is {IMP['max_mean']:.4f} AUC points and {IMP['consistent']} of {IMP['n_features']} features are positive in every fold (about {IMP['expected_chance']:.1f} expected by luck). The Week 6 univariate screen agrees ({W6['admission_factors_sig_raw']} of {W6['admission_factors_tested']} factors significant before correction, {W6['admission_factors_sig_holm']} after). "
          "A prediction-time audit found one mild look-ahead feature (PatientVisits counts the whole year) and confirmed that diagnosis, severity, labs, prescriptions and referral are post-visit information that must not be used at booking time.")
table(d, LEAK.rename(columns={"Known at prediction time?": "Known when predicting?", "In candidate?": "In candidate?"}), [5.5, 3, 3, 5.5], size=8)
d.add_heading("6.3 Can this testing method detect a real signal? (T10, T11)", 2)
para(d, f"A 'no signal' conclusion is only trustworthy if the same pipeline would have found a signal had one existed. Shuffled labels give a mean AUC of {R['negative_control']['null_mean']:.3f} (sd {R['negative_control']['null_sd']:.3f}), confirming the pipeline is unbiased. "
          f"When genuine Age effects of known strength are planted, they are recovered and flagged significant from a true AUC of about {R['min_detectable_auc']:.2f} upward, and no effect is detected when none is planted. With about 586 admissions, effects of AUC ≥ 0.57 in the booking-time data would therefore have been found.")
pl = PL.copy(); pl.columns = ["Planted effect", "True AUC of signal", "AUC recovered (CV)", "Permutation p", "Detected"]; table(d, pl, size=9)
fig(d, "w06")

d.add_heading("7. Baseline versus candidate versus refined model", 1)
d.add_heading("7.1 Refinement activities (T12)", 2)
para(d, "Each refinement addresses a finding from testing; selection used only the development set and required a gain of at least 0.02 AUC with a significant corrected resampled t-test.")
bullets(d, [("R1 Clinical-core features — ", "removes the noisy department/weekday/province/occupation categories that drove the flags."), ("R2a/R2b Stronger L2 and L1 selection — ", "addresses mild overfitting."),
            ("R3 Balanced class weights — ", "addresses the model never flagging anyone at 0.5."), ("R4 Shallow gradient boosting — ", "tests whether non-linear structure exists."),
            ("R5 Isotonic calibration — ", "addresses uninformative probabilities."), ("Rule only (RiskScore) — ", "the simplest possible benchmark.")])
v = VT[["Model", "AUC", "AUC_sd", "PR_AUC", "Brier", "dAUC_vs_candidate", "p_corrected_t"]].copy(); v.columns = ["Model", "CV AUC", "SD", "PR-AUC", "Brier", "ΔAUC vs candidate", "Corrected-t p"]; table(d, v, size=8)
para(d, f"On the development set every variant scores between {VT.AUC.min():.3f} and {VT.AUC.max():.3f}. The best refinement ({BR['name']}) gains {BR['d']:+.3f} AUC (p = {BR['p']:.2f}) — below the +0.02 threshold and not significant — and the single-rule RiskScore benchmark is as good as any model, which supports the Week 6 recommendation to use a rule rather than a model for outreach. No refinement was accepted.")
fig(d, "w07")
d.add_heading("7.2 Re-test on untouched data (T13–T15)", 2)
table(d, pd.DataFrame([["Hold-out AUC (95% CI)", "0.500", f"{CAND['auc']:.3f} ({CAND['lo']:.2f}–{CAND['hi']:.2f})", f"{REF['auc']:.3f} ({REF['lo']:.2f}–{REF['hi']:.2f})"],
                       ["Hold-out PR-AUC (prevalence 0.117)", "0.117", f"{CAND['ap']:.3f}", f"{REF['ap']:.3f}"],
                       ["Top-20% precision / recall", "0.117 (random)", f"{CAP['precision']:.3f} / {CAP['recall']:.3f}", f"{REF['precision20']:.3f} / {REF['recall20']:.3f}"],
                       ["Time-based AUC", "0.500", f"{S['time_auc']:.3f}", f"{HF['time_ref']['auc']:.3f}"],
                       ["Calibration slope", "n/a", f"{CAL['slope']:.2f}", f"{HF['calib_ref']['slope']:.2f}"]], columns=["Measure", "Baseline", "Candidate (Week 6)", f"Refined ({REF['name']})"]), [5.5, 3, 4.2, 4.3], size=9)
para(d, f"Paired on the same hold-out patients, the refined model differs from the candidate by {HF['delta']:+.3f} AUC (95% CI {HF['delta_ci'][0]:+.3f} to {HF['delta_ci'][1]:+.3f}) — no meaningful improvement. Isotonic recalibration moves the candidate's Brier score from {HF['brier_candidate']:.4f} to {HF['brier_calibrated']:.4f} (baseline {HF['brier_base']:.4f}), which is simply the average rate.")
d.add_heading("7.3 Acceptance criteria — final verdict", 2)
ct = CRT.copy(); ct.columns = [("Criterion" if c in ("Criterion", "index") else c) for c in ct.columns]; table(d, ct, [5.5, 5.75, 5.75], size=8.5, colorcol=list(ct.columns[1:]))
para(d, f"Result: the candidate meets {CR['passed'][0]} of 6 criteria and the refined model {CR['passed'][1]} of 6. Neither is suitable for patient-level use.", bold=True)

d.add_heading("8. Interpretation and model suitability", 1)
bullets(d, [("What the results mean. ", "Within these data, hospital admission cannot be predicted from booking-time information better than chance. The tests also explain why: the outcome is statistically independent of the recorded patient, doctor, department and calendar variables, the candidate fits noise, and added flexibility only memorises it."),
            ("Is the improvement meaningful? ", "No. Refinements moved AUC by at most +0.02 (not significant); the hold-out difference is +0.008 with an interval spanning zero."),
            ("Remaining weaknesses. ", "No discrimination, uninformative probabilities, flags driven by noise categories, mild look-ahead in PatientVisits, small time-based test set."),
            ("Suitable for the intended HealthConnect use case? ", "Not for patient-level decisions. It is suitable only as a test fixture for pipeline integration, with the 'use caution' flag attached to every score.")])
d.add_heading("End-to-end validation questions", 2)
table(d, pd.DataFrame([
 ["1. What component did I test?", "The Week 6 candidate admission classifier and its scoring interface."],
 ["2. What did I expect to happen?", "Meet the six pre-registered criteria; handle valid and invalid inputs as per the contract."],
 ["3. What actually happened?", f"0 of 6 criteria met (hold-out AUC {CAND['auc']:.3f}, CI {CAND['lo']:.2f}–{CAND['hi']:.2f}); interface passed {IF['v1_pass']}/{IF['total']} tests."],
 ["4. What issue or weakness did I identify?", "No predictive skill; noise-driven flags; uninformative probabilities; silent acceptance of invalid input; CSV 'None'→NaN pitfall; look-ahead feature."],
 ["5. What did I change or improve?", "Tried seven refinements (none accepted); built a validated scoring interface with an input/output contract, model card and caution flag."],
 ["6. What happened when I tested it again?", f"Model: no improvement (hold-out Δ {HF['delta']:+.3f}). Interface: {IF['v2_pass']}/{IF['total']} pass."],
 ["7. Which other track was involved?", "ML Engineering (interface hand-off; teammate sign-off pending), with Data Analytics findings reused and Project Management issue log."],
 ["8. How does the validated component contribute to HealthConnect?", "It gives the pipeline a tested, documented scoring component and an evidence-based 'no' on patient-level admission prediction, avoiding a misleading model in the final solution."],
 ["9. What still needs to be completed before Week 8?", "See section 11."]], columns=["Question", "Answer"]), [5.5, 11.5], size=9)

d.add_heading("9. Business relevance and the missed-appointment objective", 1)
para(d, "The project question is how HealthConnect can use data and AI to reduce missed appointments. Three conclusions follow from the testing:")
bullets(d, [("A no-show model cannot be built from the current data. ", "There is no attendance, cancellation or no-show field, and no booking lead time, reminder or contact history. The admission classifier is a different target and does not answer the project question."),
            ("The tested harness transfers directly. ", "Pre-registered criteria, stability and time-based splits, calibration, error and segment analysis, planted-signal power test and the interface contract are target-independent; once an attendance field is available they can be re-run in minutes."),
            ("Where the validated Week 6 findings help now. ", "Rule-based outreach to high-risk patients (33.3% meet the rule; only 1,498 of 6,662 had a 2026 appointment), the 2027 visit forecast (≈4,975 a month, ±14%) for slot planning, and the collections findings (0.92 bn RWF unpaid) remain valid and do not depend on this model.")])

d.add_heading("10. Cross-track testing and integration evidence", 1)
d.add_heading("10.1 Data Science → ML Engineering: model hand-off and scoring-interface test", 2)
para(d, f"Matrix connection: 'test whether the integrated pipeline correctly handles model requirements and expected inputs/outputs.' The Week 6 scoring behaviour (V1, a bare model call) was tested against an explicit input/output contract with 18 cases. V1 failed {n_fail_if}: impossible values (Age = -5, province = Mars) were scored without complaint, a missing ChronicCondition was silently scored as a category (the CSV 'None'→NaN pitfall, which also affected my own re-load of the Week 6 export), extra columns such as a patient name were accepted unreported, empty input crashed, error messages did not name the field, and the output was a bare array with no version or caution flag. "
          f"After the refinement (V2: contract, validation, warnings, documented output schema, caution flag) all {IF['v2_pass']} tests pass. Test → Finding → Action → Retest → Validated improvement.")
x = XT[["ID", "Scenario", "Expected", "V1", "V2"]].copy(); x.columns = ["ID", "Scenario", "Expected", "V1 (Week 6)", "V2 (refined)"]; table(d, x, [1.2, 5.7, 5.7, 2.1, 2.3], size=8, colorcol=["V1 (Week 6)", "V2 (refined)"])
fig(d, "w08", 14)
d.add_heading("10.2 Cross-track record", 2)
table(d, XREC, [4.5, 12.5], size=8.5)
box(d, "The brief states that communication alone is not sufficient and that the activity must produce Test → Finding → Action → Retest. Items 4 and 6 contain a bracketed placeholder for the ML Engineering teammate's own run of the hand-off tests and sign-off (instructions: week7/HANDOFF_TO_ML_ENGINEERING.md). Complete them with the teammate's real result before submission; do not submit them as complete until that has happened.", fill="FFF4D6", bold_first="To complete:")
d.add_heading("10.3 Other track connections", 2)
bullets(d, [("Data Analytics → Data Science. ", "The Week 6 univariate/statistical findings (0 of 26 admission factors significant after correction; admission rate not related to age, diagnosis or visit type) were used to check feature validity in the model tests; the model results agree. These were my own Week 6 analytical findings; a teammate's validated KPI findings can be added in the same table when available."),
            ("Data Science → Project Management. ", "The integration & issue log (Appendix B) lists 11 issues with severity, owner, status and retest for the PM track to maintain.")])

d.add_heading("11. Assumptions, limitations, risks and dependencies", 1)
table(d, pd.DataFrame([
 ["Issues resolved", "Interface validation (I-03); CSV 'None' guard (I-04); post-visit features excluded (I-06)."],
 ["Issues unresolved", "No predictive skill (I-01); no attendance field (I-02); uninformative probabilities (I-07); inconsistent source keys (I-08); look-ahead in PatientVisits (I-05); teammate sign-off pending (I-10)."],
 ["New issues discovered", "Silent acceptance of invalid input; CSV 'None'→NaN pitfall in exported data; noise categories driving flags; look-ahead feature."],
 ["New cross-track dependencies", "ML Engineering must adopt the contract and the 'use caution' flag; Project Management must resolve the data-source question (I-02) before Week 8."],
 ["Technical limitations", "Scikit-learn only; Jupyter/statsmodels could not be installed in the working environment, so the notebook is executed by a custom runner (standard .ipynb with saved outputs); scoring tested on 10,000 rows, not production loads."],
 ["Data limitations", "Single year (2026); 586 admissions; inconsistent facility/department/doctor keys; no attendance field."],
 ["Analytical limitations", "Time-based test has only 1,235 appointments (117 admissions in the Week 6 hold-out); segment tests have wide intervals."],
 ["Modelling limitations", "No signal means no model can be validated; refinement search was limited to seven pre-specified variants."],
 ["AI safety/reliability", "A model with no skill could mislead staff if shown as risk percentages; hence the caution flag and the recommendation not to display scores."],
 ["Integration challenges / testing limitations", "Interface tests were designed and run by Data Science; teammate verification outstanding. Contract ranges are based on the 2026 data."],
 ["Potential project impact / mitigation", "Building on this model in Week 8 would produce a misleading demo; mitigate by treating it as a fixture and re-targeting to attendance once data exist."]], columns=["Category", "Detail"]), [4.5, 12.5], size=9)

d.add_heading("12. Week 8 recommendations and outstanding actions", 1)
bullets(d, ["Resolve the data-source question: obtain the missed-appointment dataset (HealthConnect_Appointment_Data.csv with attendance, lead time, reminders) or agree in writing that the admission model is the proxy.",
            "Re-run the full test harness (notebook sections B–F) on the attendance target; apply the same pre-registered criteria.",
            "Complete the ML Engineering teammate run of the hand-off tests and record the result in the cross-track record.",
            "In the final integration, show only the rule-based high-risk list and aggregate forecasts to users; do not display model risk percentages.",
            "Recompute PatientVisits as 'visits before the appointment date'; read all exported CSVs with keep_default_na=False.",
            "Add the validated findings from Data Analytics (attendance KPIs) as candidate features; collect 24+ months of data and re-test the time-based split.",
            "Update the README, commit Week 7 and present the testing evidence in the final presentation."])

d.add_heading("Appendix A. Testing record", 1)
para(d, "Abbreviated record (all 11 fields are in HC_Week7_Testing_Validation_Record.xlsx). Result = outcome of the test as originally run; Retest = result after refinement.")
ta = tests[["ID", "Scenario", "Expected", "Actual", "Result", "Issue", "Action", "Retest"]].copy(); ta["Issue / action"] = (ta.Issue.fillna("") + " " + ta.Action.fillna("")).str.strip(); ta = ta.drop(columns=["Issue", "Action"])[["ID", "Scenario", "Expected", "Actual", "Result", "Issue / action", "Retest"]]
table(d, ta, [1.0, 3.4, 3.2, 4.2, 1.2, 3.0, 2.0], size=7, colorcol=["Result"])
d.add_heading("Appendix B. Integration & issue log", 1)
table(d, ISSUES, [1, 5, 1.6, 2, 2, 2, 3.2, 2], size=7)
d.add_heading("Appendix C. Reproducing this work", 1)
bullets(d, ["Repository folder week7/: src/week7_model_testing.py (analysis notebook source), src/admission_risk_scorer.py (scoring wrapper, V1 and V2), src/build_week7_docs.py (this report).",
            "Run order: Week 6 scripts 01–06 (creates outputs/data/HC_Integrated_Encounters_5000.csv), then python week7/src/_nb_runner.py week7/src/week7_model_testing.py week7/outputs/HC_Week7_Model_Testing_Notebook.ipynb, then python week7/src/build_week7_docs.py.",
            "Hand-off files: outputs/model_card.md, outputs/input_output_contract.json, outputs/models/admission_scorer_v2.joblib, outputs/tables/t_scoring_interface_tests.csv."])
d.save("outputs/reports/HC_Week7_Model_Testing_and_Refinement_Report.docx"); print("saved report")

# ------------------------------------------------------------------ PROJECT SUMMARY ------------------------------------------------------------------
d = new_doc("Week 7 Project Summary", "Data Science track — complements (does not repeat) the Model Testing & Refinement Report")
rows = [
 ["1. What I planned to test", "The Week 6 candidate admission classifier (stability, metrics, calibration, errors, segments, overfitting, feature validity) and its scoring interface for ML Engineering."],
 ["2. What I actually tested", f"{TS['n']} recorded tests: 15 model tests and 18 interface tests, against six acceptance criteria fixed in advance; hold-out opened once."],
 ["3. Most important results", f"Hold-out AUC {CAND['auc']:.3f} (CI {CAND['lo']:.2f}–{CAND['hi']:.2f}); 0 of 6 criteria met; calibration slope {CAL['slope']:.2f}; planted signals ≥ AUC {R['min_detectable_auc']:.2f} are detectable, so the result is a data finding."],
 ["4. Issues or weaknesses", "No predictive skill; flags driven by noise categories; uninformative probabilities; silent acceptance of invalid inputs; CSV 'None'→NaN pitfall; look-ahead in PatientVisits; no attendance field in the data."],
 ["5. Improvements made", f"Seven model refinements tried (none accepted); scoring interface rebuilt with an input/output contract, validation, output schema, model card and 'use caution' flag."],
 ["6. Retesting results", f"Model: hold-out Δ AUC {HF['delta']:+.3f} (CI {HF['delta_ci'][0]:+.2f} to {HF['delta_ci'][1]:+.2f}), still 0/6 criteria. Interface: {IF['v1_pass']}/{IF['total']} → {IF['v2_pass']}/{IF['total']} pass."],
 ["7. Tracks collaborated with", "ML Engineering (hand-off prepared; teammate sign-off pending); Data Analytics findings reused; Project Management issue log."],
 ["8. What was tested collaboratively", "Model scoring interface: valid, invalid, missing, out-of-range, unseen and empty inputs; output schema; persistence; speed."],
 ["9. What changed as a result", "Invalid input now produces a clear, field-named error or warning; outputs carry a version and a caution flag; identifiers in extra columns are reported."],
 ["10. Key findings / validation outcomes", "The model is not suitable for patient-level use; the test harness itself is validated (negative control and planted-signal power test) and reusable."],
 ["11. Major challenges", "Brief assumes a no-show dataset the project does not have; no signal in the data; Jupyter/statsmodels not installable in the working environment."],
 ["12. Important decisions made", "Test the Week 6 admission model rather than switch dataset (per the brief); pre-register criteria; reject refinements that do not clear +0.02 AUC with a significant corrected test; keep the model as a fixture only."],
 ["13. Remaining limitations", "One year of data; 586 admissions; inconsistent source keys; interface tests not yet verified by the ML Engineering teammate."],
 ["14. Remaining issues / dependencies", "I-01 (no skill), I-02 (attendance data source), I-05 (look-ahead feature), I-07 (probabilities), I-08 (source keys), I-10 (teammate sign-off) — see issue log."],
 ["15. Contribution to HealthConnect", "An evidence-based decision that avoids a misleading model, a validated and reusable testing harness, and a documented scoring component ready for pipeline integration."],
 ["16. To complete before Week 8", "Obtain the attendance dataset (or confirm the proxy), re-run the harness on that target, complete the teammate's hand-off test, drop model risk percentages from any user-facing output, commit and present."]]
table(d, pd.DataFrame(rows, columns=["Item", "Summary"]), [4.5, 12.5], size=9)
d.save("outputs/reports/HC_Week7_Project_Summary.docx"); print("saved summary")
