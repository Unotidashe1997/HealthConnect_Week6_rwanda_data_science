"""Week 8 builder: final package docx, one-page summary, video script, case study, requirements md, integration workbook.
Run from anywhere:  python src/build_week8_docs.py   (after src/week8_final_model_evaluation.py)"""
import json, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _docx_helpers import *
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RP = "outputs/reports/"; os.makedirs(RP, exist_ok=True)
R = json.load(open("outputs/results.json"))["R"]; W7 = json.load(open("../week7/outputs/results.json"))["R"]
HF, CR, IF, CC, FV, S = W7["holdout_final"], W7["criteria"], W7["interface"], W7["conf_counts"], W7["flag_vs_unflag"], W7["stability"]
RU, HR = R["rule"], R["highrisk"]
F1, F2 = "outputs/figures/w8_01_baseline_vs_final.png", "outputs/figures/w8_02_integration_view.png"
T = lambda n: pd.DataFrame(R[n])
PH = "[TO COMPLETE with teammate]"

# ---------------------------------------------------------------- content shared by several documents
TRANS = [
 ("Main Week 7 output", "A tested and refined model package: a six-criterion test of the Week 6 admission model, seven refinements, and a hardened scoring interface (V2) with a model card and hand-off note."),
 ("Most important testing result", f"The model has no usable predictive skill: hold-out AUC {HF['cand']['auc']:.2f} (95% CI {HF['cand']['lo']:.2f}–{HF['cand']['hi']:.2f}), 0 of 6 acceptance criteria met. Planted-signal tests proved the method would have found real signal."),
 ("Major issue found", f"The data has no attendance/no-show field, so the project target (missed appointments) cannot be modelled; the interface also failed {IF['total']-IF['v1_pass']} of {IF['total']} tests, and PatientVisits looks ahead."),
 ("What was refined", f"Seven feature/model refinements (none accepted) and the scoring interface: validated inputs, clear errors, risk bands, caution flag, versioning. Interface tests {IF['v1_pass']}/{IF['total']} → {IF['v2_pass']}/{IF['total']}."),
 ("Component ready for integration", "The V2 scoring interface, input/output contract, model card and testing harness (as an integration fixture). Not the model's predictions."),
 ("Still to integrate", "Validated attendance KPIs from Data Analytics; ML Engineering pipeline confirmation; GenAI scope; PM risk/readiness log. The attendance target cannot be added without an attendance dataset."),
 ("Tracks involved", "Data Analytics, Data Science (me), ML Engineering, GenAI, Project Management."),
 ("What to demonstrate", "Baseline versus final comparison, honest error analysis, the can/cannot-use matrix, and the rule-based outreach list and forecast as the usable outputs.")]
READY = [
 ("Final component", "Final Model & Data Science Package: Week 6 admission model (unchanged logistic regression) kept as a tested integration fixture, plus testing harness, scoring interface V2, rule-based risk list and forecasts."),
 ("Problem addressed", "How HealthConnect can use data and AI to reduce missed appointments; what a model can and cannot responsibly support given the available data."),
 ("Current status", "Complete and documented. Model status: NOT approved for patient-level decisions (0/6 criteria). Interface and harness: ready."),
 ("What improved in Week 7", f"Systematic testing; refinement trials; interface hardening ({IF['v1_pass']}/{IF['total']} → {IF['v2_pass']}/{IF['total']} tests); model card; issue log."),
 ("Tracks it depends on", "Data Analytics (validated KPIs and data-quality findings); Project Management (scope and priorities); a future attendance dataset."),
 ("Tracks that depend on it", "ML Engineering (scorer, contract, requirements); GenAI (only the plain-language model limits); Project Management (risks, readiness)."),
 ("Output provided", "Scorer V2 (joblib), input/output contract, model card, FINAL_MODEL_REQUIREMENTS.md, test harness, risk list rule, 2027 forecasts, this package."),
 ("Output needed from other tracks", "DA: validated attendance/no-show KPIs and data dictionary; MLE: pipeline run confirmation; PM: sign-off on scope and risk log. (Not yet received unless confirmed below.)"),
 ("Evidence of readiness", f"Interface tests {IF['v2_pass']}/{IF['total']}; 6 pre-registered criteria evaluated; negative control and planted-signal tests; notebooks, tables and figures in the GitHub repository."),
 ("Limitations", "No predictive skill; wrong target (admission, not no-show); one year of data; PatientVisits look-ahead; no external validation; teammate integration evidence still to be attached.")]
WALK = [
 ("1. Problem", "HealthConnect Clinic loses capacity and patient trust when appointments are missed; it needs data and AI to reduce this and improve patient support."),
 ("2. Data", "27 files; 20,000 patients, 5,000 encounters, 60,000 bills. No attendance field, so missed appointments cannot be measured or modelled directly."),
 ("3. Analytics", f"Stable ≈4,975 visits/month; {HR['n']:,} high-risk patients (33.3%), only {HR['seen']:,} with a 2026 appointment; weak billing and claims control (≈0.92 bn RWF unpaid, 21,028 pending claims). 0 of 26 admission factors significant."),
 ("4. Predictive component (Data Science)", f"Tested admission model: AUC {HF['cand']['auc']:.2f}, no gain over baseline. Rule-based list and flat forecast are the usable outputs."),
 ("5. ML Engineering component", "Validated scorer V2 with contract, error handling and versioning, ready to be wrapped in a pipeline/API. [Teammate to confirm how it was run.]"),
 ("6. GenAI component", "Assistant answers only from the approved Knowledge Base, and may explain model limits in plain language but must not give risk predictions. [Teammate to confirm.]"),
 ("7. PM layer", "Priorities, risks, decisions and readiness tracking across tracks; the model's 'not approved' status is a recorded risk decision. [Teammate to confirm.]"),
 ("8. Final solution", "A decision-support package: outreach list for high-risk patients without a visit, slot planning from the forecast, collections follow-up, and a ready test harness for an attendance model."),
 ("9. Business value", f"Targets the {HR['unseen']:,} high-risk patients with no 2026 visit; avoids deploying a model that would mislead; protects revenue follow-up; sets up a defensible attendance model."),
 ("10. Limitations", "Model has no skill; attendance data missing; look-ahead feature; single year; rule enrichment is weak. Next step: obtain appointment-status data and retrain with the same tests.")]
XREC = [
 ("Data Analytics → Data Science", "Validated KPIs, data-quality findings, attendance patterns", "Cleaned feature definitions; confirm which patterns are real before modelling", "Findings → model interpretation (T_FIND table); fixed inconsistent keys", "Model interpretation now cites validated findings; explains why no factor predicts admission"),
 ("Data Science → ML Engineering", "Scorer V2, contract, requirements, model card", "Pipeline wrapping the scorer; confirm test run", "Requirements file; retraining and look-ahead rules", "ML pipeline receives a validated interface and a clear 'not for patient decisions' flag"),
 ("ML Engineering → Data Science", "Pipeline feedback, input errors found in testing", "Fixes to the contract after pipeline run", "Contract updates, if any", "Contract based on real pipeline behaviour"),
 ("Data Analytics → Project Management", "Validated findings and KPI definitions", "Priority list of insights for the final presentation", "Status of each finding", "PM plan uses validated, not provisional, insights"),
 ("GenAI → Project Management", "Assistant scope, Knowledge Base coverage, tested questions", "Risk log for AI answers", "Scope statement", "PM records GenAI limits and escalation rules"),
 ("Project Management → All tracks", "Priorities, risks, decisions, readiness status", "Common integration timeline", "Final status matrix", "One consistent story across all tracks")]
def xrec_df():
    rows = []
    for link, dep, rec, act, imp in XREC:
        mine = link.startswith("Data Science →")
        rows.append({"Track collaborated with": link, "Dependency": dep,
            "Output received": PH + f" — expected: {rec}" if not mine else "Feedback on scorer/requirements " + PH,
            "Output provided": "Scorer V2, contract, requirements, model card, limits statement" if mine else PH,
            "Final integration activity": act, "What changed": PH + " (record the real change)", "How it improved the solution": imp,
            "Evidence": PH + " (message, screenshot, shared file, meeting note)", "Contribution to final presentation": "Model section: baseline vs final, limits, can/cannot-use" if "Data Science" in link else PH})
    return pd.DataFrame(rows)
CONTRIB = f"""The model section of the final presentation has five points: the baseline vs final result (AUC {HF['cand']['auc']:.2f}, 0/6 criteria); the error analysis; the can/cannot-use matrix; the rule-based outreach list ({HR['unseen']:,} high-risk patients without a 2026 visit); and the requirements passed to ML Engineering."""

# ================================================================== 1. MAIN PACKAGE
d = new_doc("HealthConnect Final Model & Data Science Package", "Final model decision, evaluation, documentation and cross-track contribution")
box(d, f"The final model is the unchanged Week 6 logistic regression, kept as a tested integration fixture. It is not approved for patient-level decisions: hold-out AUC {HF['cand']['auc']:.2f} (95% CI {HF['cand']['lo']:.2f}–{HF['cand']['hi']:.2f}), 0 of 6 acceptance criteria met. The useful, validated outputs are the rule-based outreach list, the flat visit forecast, the collections findings, and a testing harness ready for an attendance model.", bold_first="Final verdict.")
d.add_heading("1. Week 7 → Week 8 transition", 1)
table(d, pd.DataFrame(TRANS, columns=["Item", "Summary"]), [4, 13], 9)
d.add_heading("2. Part 1 — Integration readiness", 1)
table(d, pd.DataFrame(READY, columns=["Field", "Entry"]), [4, 13], 9)
d.add_heading("3. Review of the Week 7 results", 1); table(d, T("review"), [5.5, 5.5, 6], 8.5)
d.add_heading("4. Final candidate confirmation", 1)
para(d, "Decision rule: choose an option that meets the acceptance criteria set in Week 7; if none does, keep the simplest, best-documented model only as an integration fixture and say so.")
table(d, T("options"), [4.2, 2, 2, 5.6, 2.6], 8)
d.add_heading("5. Baseline versus final comparison", 1)
fig(d, F1, "Figure 1. Final model versus baseline")
table(d, T("final").rename(columns={"Lo": "Lower 95%", "Hi": "Upper 95%"}), [6, 1.8, 2, 2, 5.2], 8.5)
para(d, f"The transparent RiskScore rule is a weak signal, not a strong one: on all 5,000 appointments the high-risk flag ({RU['hr_share']:.1%} of appointments) has an admission rate of {RU['hr_rate']:.1%} against {RU['non_rate']:.1%} for the rest (chi-square p = {RU['p']:.3f}), but its AUC of {RU['auc']:.2f} has an interval ({RU['lo']:.2f}–{RU['hi']:.2f}) that touches chance. It is therefore justified for outreach on clinical and equity grounds (older, chronic or underweight/obese patients), not as a predictor.")
d.add_heading("6. Error analysis summary", 1); table(d, T("errors"), [5, 6, 6], 8.5)
d.add_heading("7. Strengths and weaknesses", 1); table(d, T("strength_weak"), [3, 14], 9)
d.add_heading("8. Feature and model decision log", 1); table(d, T("decisions"), [1, 1.5, 5.5, 5, 4], 8)
d.add_heading("9. Business interpretation and suitability", 1)
para(d, "What the model can and cannot be used for:", bold=True); table(d, T("use"), [6.5, 3, 7.5], 8.5)
para(d, "How validated findings shape the interpretation:", bold=True); table(d, T("integration"), [8, 9], 8.5)
para(d, "Data Analytics teammate: add their validated attendance/no-show KPIs here when received. " + PH, italic=True)
d.add_heading("10. Limitations and risks", 1)
table(d, pd.DataFrame([
 ["Wrong target", "Admission modelled, not no-show; the dataset has no attendance field", "High", "Obtain appointment-status data; retrain with the same tests"],
 ["No predictive skill", f"AUC {HF['cand']['auc']:.2f}; probabilities uninformative (calibration slope {W7['calibration']['slope']:.2f})", "High", "Do not show scores to staff or patients; caution flag in every output"],
 ["Look-ahead feature", "PatientVisits counts the full year", "Medium", "Recompute as visits before each appointment"],
 ["Single year, no external validation", "One year of one clinic network", "Medium", "Re-test on a later period before any use"],
 ["Weak rule enrichment", f"Lift {RU['hr_rate']/RU['prev']:.2f}×; AUC interval touches chance", "Medium", "Use rule for outreach only; monitor results"],
 ["Data quality", "Inconsistent keys, unreliable bill status", "Medium", "Fix at source (Analytics recommendation)"],
 ["Privacy", "Patient-level data and names", "High", "Names/phones excluded; patient-level data not pushed to GitHub"]],
 columns=["Risk", "Detail", "Severity", "Mitigation"]), [3.2, 6, 1.8, 6], 8.5)
d.add_heading("11. Requirements for ML Engineering", 1)
bullets(d, ["Use scorer V2 with the input/output contract (18 fields, ranges, categories); reject invalid input with the field-named errors.",
 "Read CSVs with keep_default_na=False, na_values=[''] — the value 'None' (no chronic disease) is real data, not missing.",
 "Recompute PatientVisits as visits before the appointment date before any production use.",
 "Return risk_band, model_version and use_caution with every score; never show a bare probability.",
 "Do not deploy for patient-level decisions. Retrain on an attendance target once available and rerun all six criteria.",
 "Full detail: FINAL_MODEL_REQUIREMENTS.md and week7/outputs/model_card.md."])
d.add_heading("12. Final cross-track integration (HC-POD records)", 1)
para(d, "Each link below has the nine fields required. Entries marked [TO COMPLETE] must be filled by the intern with the real details and evidence from the teammate; I have not claimed any collaboration that was not recorded.", italic=True)
for _, r in xrec_df().iterrows():
    d.add_heading(r["Track collaborated with"], 3); table(d, pd.DataFrame({"Field": r.index[1:], "Entry": r.values[1:]}), [4.5, 12.5], 8.5)
fig(d, F2, "Figure 2. Data Science in the integrated solution")
d.add_heading("13. End-to-end solution walkthrough", 1); table(d, pd.DataFrame(WALK, columns=["Step", "Explanation"]), [4, 13], 9)
d.add_heading("14. Contribution to the final presentation", 1); para(d, CONTRIB)
d.add_heading("15. Reflection", 1)
bullets(d, ["A failed model test is a finding, not a failure of the work: proving that the data holds no usable signal prevented a misleading deployment.",
 "Testing the testing (planted signals, shuffled labels) made the negative result defensible.",
 "Target mismatch should be checked on day one: the project asked about missed appointments, the data could only support admissions.",
 "The most valuable outputs were simple: a transparent rule, a flat forecast, and a clear can/cannot-use statement."])
d.save(RP + "HC_Week8_Final_Model_and_DataScience_Package.docx")

# ================================================================== 2. ONE-PAGE NON-TECHNICAL SUMMARY
d = new_doc("HealthConnect: Model Summary in Plain Language", "One page for non-technical readers")
para(d, "What we tried", bold=True); para(d, "We tested whether a computer model could tell, at booking time, which patients are more likely to be admitted to hospital. We used it as a stand-in because the data does not record missed appointments.")
para(d, "What we found", bold=True); para(d, f"The model was no better than guessing. Across {HF['cand']['n'] if 'n' in HF['cand'] else 1000:,} unseen appointments its score was {HF['cand']['auc']:.2f} where 0.50 is a coin toss and 0.65 was our minimum. We checked this in many ways, including tests that planted a real signal to show we would have found it.")
para(d, "What this means", bold=True)
bullets(d, ["Do not use this model to decide anything about an individual patient.", "Do not show staff or patients a risk percentage from it.", "The model must not be used to predict missed appointments: that data does not exist yet."])
para(d, "What we can use now", bold=True)
bullets(d, [f"A simple, transparent list of {HR['n']:,} higher-risk patients (older, chronic disease, or weight extremes). {HR['unseen']:,} of them had no appointment in 2026, so outreach could help them.", "A steady forecast of about 4,975 visits per month for planning slots.", "Collections follow-up on unpaid bills and pending claims."])
para(d, "What we need next", bold=True); para(d, "Appointment status records (attended, missed, cancelled). With them the same tests can show quickly whether a no-show model works.")
d.save(RP + "HC_Week8_Plain_Language_Model_Summary.docx")

# ================================================================== 3. VIDEO SCRIPT
d = new_doc("Individual Presentation Video Script", "5–10 minute video — to be recorded and presented personally by the intern (about 7½ minutes)")
box(d, "Read this in your own words, on camera, with your voice. Replace every [bracket] with your real details. Describe only what you personally did; for teammates only say what they confirmed.", bold_first="Before you record.")
SEC = [("1. Introduction (0:00–0:45)", f"Hello, I'm [your name], an AnalystLab Africa Experience Lab intern on the Data Science track. Our project is HealthConnect Clinic, and the question we worked on is: how can the clinic use data and AI to reduce missed appointments and improve the patient support experience? In this video I'll show my contribution, how it connected to my teammates, what testing showed, and the final outcome."),
 ("2. Track contribution (0:45–3:15)", f"My job was to build and judge the predictive model. The data has 20,000 patients and 5,000 appointments, but no field recording whether someone attended. So I modelled hospital admission as a stand-in, using only information known at booking. In Week 7 I tested that model against six acceptance criteria I set in advance. On unseen data its AUC was {HF['cand']['auc']:.2f}. A score of 0.50 is a coin toss, and my minimum was 0.65. It met none of the six criteria. I then tried seven refinements, and the best moved the score by under one point, which is not a real gain. To be sure the test itself was fair, I planted known signals into the data and the method found them, and when I shuffled the answers the score fell to chance. So the finding is honest: this data holds no usable signal for this model. My final decision was to keep the model only as a test fixture and mark it not approved for patient decisions. What I recommended instead is a transparent rule: {HR['n']:,} higher-risk patients, of whom {HR['unseen']:,} had no appointment in 2026, plus a steady forecast of about 4,975 visits a month."),
 ("3. Cross-track collaboration (3:15–4:45)", "I worked with [Data Analytics teammate] who [what they gave me]. That helped me [how it changed the model interpretation]. I gave [ML Engineering teammate] the scoring interface, the contract of allowed inputs, and a requirements file, and they [what they did / confirmed]. I also shared the model limits with [GenAI teammate] and [PM teammate], who [what they recorded]. [Replace this paragraph with only what really happened.]"),
 ("4. Testing, refinement and final outcome (4:45–6:15)", f"Testing found two serious issues. First, the target mismatch, no attendance data. Second, my scoring interface passed only {IF['v1_pass']} of {IF['total']} tests: it accepted bad inputs silently. I rebuilt it with validation, clear errors, risk bands and a caution flag, and it now passes {IF['v2_pass']} of {IF['total']}. The final comparison: baseline 0.50, my model {HF['cand']['auc']:.2f}, best refinement {HF['ref']['auc']:.2f}. None reaches 0.65."),
 ("5. Overall solution (6:15–7:15)", "Putting it together: Analytics shows where the problems are, Data Science tells us what the data can and can't predict, ML Engineering makes the pipeline reliable, GenAI answers patient questions from the approved Knowledge Base, and Project Management keeps the risks and decisions in one place. The business value is targeted outreach, better slot planning and honest limits, so nobody trusts a risk score that doesn't work."),
 ("6. Closing (7:15–8:00)", "My key lesson: a good test that says 'no' is more useful than a model that says 'yes' wrongly. The next step is to get appointment status data and rerun the same tests. Thank you for watching. I'm [your name], and the code and documentation are on my GitHub: [your repo link].")]
for h_, t_ in SEC: d.add_heading(h_, 2); para(d, t_)
para(d, "Checklist: face and voice on camera; screen share of slides 6–9; stay between 5 and 10 minutes; do not claim anything a teammate has not confirmed.", italic=True)
d.save(RP + "HC_Week8_Video_Script.docx")

# ================================================================== 4. CASE STUDY
d = new_doc("Portfolio Case Study: Knowing When a Model Should Not Be Used", "HealthConnect Clinic — AnalystLab Africa Experience Lab, Data Science track")
for h_, t_ in [("Context", "HealthConnect Clinic wanted to use data and AI to reduce missed appointments and improve patient support. I worked on the Data Science track, alongside Data Analytics, ML Engineering, GenAI and Project Management."),
 ("Challenge", "The dataset (20,000 patients, 5,000 encounters, 60,000 bills) had no attendance field, so missed appointments could not be modelled directly."),
 ("What I did", f"Cleaned and integrated the data; built an 8-script pipeline; tested a logistic regression admission model against six pre-set criteria; ran seven refinements; validated the method with planted signals and shuffled labels; rebuilt the scoring interface (tests {IF['v1_pass']}/{IF['total']} → {IF['v2_pass']}/{IF['total']}); wrote requirements and a model card for ML Engineering."),
 ("Result", f"The model had no usable skill (AUC {HF['cand']['auc']:.2f}, interval {HF['cand']['lo']:.2f}–{HF['cand']['hi']:.2f}; 0/6 criteria), and I reported that plainly. The validated outputs were a transparent outreach list ({HR['n']:,} high-risk patients, {HR['unseen']:,} with no 2026 visit), a flat forecast (≈4,975 visits/month) and collections findings."),
 ("Impact", "Prevented the deployment of a misleading risk score, pointed outreach at the patients who need it, and left a reusable test harness for an attendance model."),
 ("Tools", "Python, pandas, scikit-learn, scipy, matplotlib, python-docx, python-pptx, GitHub."),
 ("Lessons learned", "Check that the target exists before modelling. Test your tests. A negative result with evidence is a deliverable. Keep model limits written next to every output.")]:
    d.add_heading(h_, 2); para(d, t_)
d.add_heading("Links", 2); para(d, "GitHub: https://github.com/Unotidashe1997/healthconnect_week6_rwanda_data_science")
d.save(RP + "HC_Week8_Portfolio_Case_Study.docx")

# ================================================================== 5. MD DOCS
open("FINAL_MODEL_REQUIREMENTS.md", "w").write(f"""# Final model requirements — hand-off to ML Engineering (Week 8)

**Status:** Week 6 logistic regression (L2, C=0.1, 18 features), unchanged. Integration fixture only. **Not approved for patient-level decisions** (hold-out AUC {HF['cand']['auc']:.3f}, 0/6 criteria).

## Interface
- Use `week7/src/admission_risk_scorer.py` (`AdmissionScorerV2`, version `week7-v2`) and `week7/outputs/input_output_contract.json`.
- Inputs: 18 required fields; numeric ranges and category lists per the contract; invalid input raises `ScoringError` naming the field.
- Outputs per row: `risk_score`, `risk_band`, `warnings`, `model_version`, `use_caution` (always True at present).

## Data handling
- Read CSVs with `keep_default_na=False, na_values=[""]` (`ChronicCondition == "None"` is a real value).
- `PatientVisits` currently counts the whole year (look-ahead). Recompute as visits *before* the appointment.
- Never log or export names, phone numbers or emergency contacts.

## Do / do not
- Do: use for pipeline, API and logging tests; display the caution flag.
- Do not: show scores to staff or patients; use for individual decisions; use as a no-show predictor.

## Retraining plan
1. Obtain appointment status (attended / missed / cancelled).
2. Rebuild the target, recompute booking-time features.
3. Rerun the six criteria, the time-split test, calibration, and the planted-signal check (`week7/src/week7_model_testing.py`).
4. Deploy only if all criteria pass.

## Dependencies
Python 3.11+, pandas, numpy, scikit-learn 1.9.1, scipy, joblib (see `requirements.txt`). The joblib model must be loaded with the same scikit-learn version.
""")
pd.ExcelWriter
with pd.ExcelWriter(RP + "HC_Week8_Integration_Records.xlsx", engine="xlsxwriter") as xw:
    xrec_df().to_excel(xw, sheet_name="HC-POD records", index=False)
    pd.DataFrame(READY, columns=["Field", "Entry"]).to_excel(xw, sheet_name="Readiness", index=False)
    pd.DataFrame(WALK, columns=["Step", "Explanation"]).to_excel(xw, sheet_name="Walkthrough", index=False)
    T("decisions").to_excel(xw, sheet_name="Decision log", index=False); T("use").to_excel(xw, sheet_name="Can-Cannot use", index=False)
    for n_ in xw.sheets: xw.sheets[n_].set_column(0, 9, 36)
print("built")
