"""Builds the Data Science section of the final presentation (HC_Week8_DataScience_Final_Presentation.pptx)."""
import json, os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
R = json.load(open("outputs/results.json"))["R"]; W7 = json.load(open("../week7/outputs/results.json"))["R"]
HF, IF, RU, HR = W7["holdout_final"], W7["interface"], R["rule"], R["highrisk"]
BLUE, INK, GREY, WHITE, AMB = RGBColor(0x2a,0x78,0xd6), RGBColor(0x11,0x11,0x11), RGBColor(0x66,0x66,0x66), RGBColor(255,255,255), RGBColor(0xeb,0x68,0x34)
P = Presentation(); P.slide_width, P.slide_height = Inches(13.333), Inches(7.5)
def tb(s, x, y, w, h, text, size=18, bold=False, color=INK):
    t = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); f = t.text_frame; f.word_wrap = True
    for i, line in enumerate(text if isinstance(text, list) else [text]):
        p = f.paragraphs[0] if i == 0 else f.add_paragraph(); p.text = line; p.font.size = Pt(size); p.font.bold = bold; p.font.color.rgb = color; p.space_after = Pt(8)
    return t
def slide(title, bullets=None, img=None, notes="", kicker=None, size=20):
    s = P.slides.add_slide(P.slide_layouts[6]); bar = s.shapes.add_shape(1, 0, 0, P.slide_width, Inches(0.18)); bar.fill.solid(); bar.fill.fore_color.rgb = BLUE; bar.line.fill.background()
    tb(s, .6, .4, 12, 1, title, 32, True, BLUE)
    if kicker: tb(s, .6, 1.3, 12, .8, kicker, 20, True, AMB)
    y = 2.1 if kicker else 1.5
    if bullets: tb(s, .6, y, 5.8 if img else 12, 5, ["• " + b for b in bullets], size)
    if img: s.shapes.add_picture(img, Inches(6.6 if bullets else 1.2), Inches(y), width=Inches(6.3 if bullets else 11))
    tb(s, .6, 7.0, 12, .4, "HealthConnect Clinic · AnalystLab Africa Experience Lab · Week 8 · Data Science", 10, False, GREY)
    s.notes_slide.notes_text_frame.text = notes; return s
s = P.slides.add_slide(P.slide_layouts[6]); bg = s.shapes.add_shape(1, 0, 0, P.slide_width, P.slide_height); bg.fill.solid(); bg.fill.fore_color.rgb = BLUE; bg.line.fill.background()
tb(s, .8, 2.2, 11.5, 1.5, "HealthConnect Final Model & Data Science Package", 40, True, WHITE); tb(s, .8, 3.9, 11.5, 1, "What the data can and cannot predict, and what we recommend instead", 22, False, WHITE); tb(s, .8, 5.5, 11.5, 1, "[Your name] · Data Science track · AnalystLab Africa Experience Lab, Week 8", 16, False, WHITE)
slide("The problem and my role", ["Project question: how can HealthConnect use data and AI to reduce missed appointments and improve patient support?", "My track: Data Science — build and judge the predictive model, set what it may be used for.", "Key constraint: the data has no attendance field, so I modelled hospital admission as a stand-in."], notes="Introduce yourself, then the question and the constraint.")
slide("The data and what Analytics showed", ["20,000 patients · 5,000 appointments · 60,000 bills", "Visits stable at ≈4,975 a month, no trend", f"{HR['n']:,} high-risk patients (33.3%); only {HR['seen']:,} had a 2026 appointment", "≈0.92 bn RWF unpaid; 21,028 pending claims", "0 of 26 admission factors significant after correction"], notes="These are the validated findings from my Week 6 analysis; add your Analytics teammate's KPIs here.")
slide("How the model was tested", ["Six acceptance criteria fixed before testing (AUC ≥ 0.65, interval above 0.5, calibration, stability, gain over baseline…)", "Untouched hold-out of 1,000 appointments; time split Oct–Dec", "Planted-signal and shuffled-label checks to prove the method works", "Seven refinements tried on development data only"], notes="Emphasise the criteria were set in advance.")
slide("Result: no better than chance", [f"Hold-out AUC {HF['cand']['auc']:.2f} (CI {HF['cand']['lo']:.2f}–{HF['cand']['hi']:.2f})", f"Best refinement {HF['ref']['auc']:.2f}: not a real gain", "0 of 6 criteria met", "The test would have found a real signal from AUC ≈ 0.57"], img="outputs/figures/w8_01_baseline_vs_final.png", notes="Point at the dashed 0.50 and dotted 0.65 lines.", size=18)
slide("Error analysis", [f"Flag top 20%: admission rate {W7['flag_vs_unflag']['flagged_rate']:.1%} vs {W7['flag_vs_unflag']['unflagged_rate']:.1%} unflagged (p = {W7['flag_vs_unflag']['p']:.2f})", f"{W7['conf_counts']['FN']} of {W7['conf_counts']['TP']+W7['conf_counts']['FN']} admissions missed", "Flags follow department/weekday/province categories: noise, not outcome", "No segment where the model works"], notes="The model's flags do not enrich for admissions.")
slide("Final model decision", ["Kept: Week 6 logistic regression, unchanged", "Status: integration fixture. NOT approved for patient-level decisions", "Rejected: complex models (memorise noise), refinements (no gain)", f"Interface hardened: tests {IF['v1_pass']}/{IF['total']} → {IF['v2_pass']}/{IF['total']}"], kicker="A negative result, proven, is a deliverable.", notes="Explain the decision rule.")
slide("What it can and cannot be used for", ["CANNOT: rank patients, show risk %, predict no-shows, rank facilities", "CAN: test the ML pipeline end to end; reuse the test harness", "CAN (rule, not model): outreach to high-risk patients with no visit", "CAN: flat forecast for slot planning"], notes="This is the business suitability slide.")
slide("Integration across tracks", ["Analytics → DS: validated findings [confirm]", "DS → ML Engineering: scorer, contract, requirements", "ML Engineering → DS: pipeline feedback [confirm]", "GenAI, PM: limits and risks [confirm]"], img="outputs/figures/w8_02_integration_view.png", notes="Fill in what teammates actually confirmed.", size=18)
slide("Business value and recommendations", [f"Outreach to the {HR['unseen']:,} high-risk patients with no 2026 visit", "Plan slots from a steady 4,975/month forecast", "Follow up unpaid bills and pending claims", "Collect appointment status to build the real no-show model"], notes="Link back to the project question.")
slide("Limitations, risks and next steps", ["No predictive skill; wrong target; one year; look-ahead feature", "Rule enrichment is weak: admission rate 13.2% vs 11.0%", "Next: get attendance data, retrain, rerun all six criteria", "Hand-off: FINAL_MODEL_REQUIREMENTS.md"], notes="Close with the lesson learned.")
P.save("outputs/reports/HC_Week8_DataScience_Final_Presentation.pptx"); print("ok")
