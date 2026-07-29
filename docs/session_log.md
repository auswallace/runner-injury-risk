# Session log

Running narrative of what happened and why, for picking work back up cold. CLAUDE.md
has the current state; this file has the reasoning trail, including the wrong turns.
Newest session first.

## Session 3 (2026-07-29): 04_evaluation - the one-shot test run

The locked test sets were touched for the first and only time. Four open decisions
were surfaced with validation-side evidence and settled by Austin BEFORE any test
contact (recorded in assumptions_limitations.md under "Evaluation decisions"):

1. TA final model fit on train only, validation spent once on the calibrator (the
   train+val refit was rejected: its calibrator would map a different model's score
   distribution). Grouped: fit on all dev, calibrator on cross-fitted OOF preds.
2. Platt over isotonic, both splits. Cross-fitted val Brier was a near-tie (TA
   0.01527 vs 0.01532; GR 0.01391 vs 0.01379); Platt has 2 params at 132 calibration
   positives and matches the paper's own choice.
3. Alert budgets 1/2/3/5/10 flags/week, thresholds frozen on validation.
4. Median subgroup splits, dev-side cutoffs, evaluated on the grouped test.

### Results (final - these numbers do not get revisited)

- Headline: **test AP 0.030 both splits** (1.7x chance TA, 2.6x GR); **AUC 0.629 TA
  / 0.619 GR** vs paper 0.724 and our grouped CV 0.690.
- The grouped CV -> test drop is athlete heterogeneity, not a bug: per-athlete AUC
  spans 0.27-0.83, 2 of 15 evaluable test athletes below 0.5. Check (c) of the
  evaluation design caught exactly what it was built to catch.
- Calibration: Platt repairs gross miscalibration (Brier 0.245 -> 0.0173) but ties
  the base-rate reference. Ranking has value; sharpness does not. Bands, not
  percentages, in the dashboard.
- Alert budget: 2.9-4.3% precision at 1-5 flags/week, recall 2-18% - the paper's
  implied ~3% precision made explicit. Frozen thresholds under-flagged the grouped
  test (lower-prevalence pool): budgets drift across populations.
- Subgroups: better for short-tenure / low-volume athletes (AUC 0.66-0.67 vs
  0.60-0.62; small bins, direction not precision).
- w6 lead-time check: two days of warning costs ~nothing (GR AP 0.032 vs 0.030).

"When NOT to trust the score" in assumptions_limitations.md is now filled with the
five concrete blind spots above. `data/processed/evaluation_results.json` carries
every table for the dashboard layer.

### Decision-support dashboard (same session)

Austin picked self-contained HTML over Tableau Public (Tableau depth is already on
the resume; a single reviewable file shows range) and the time-aware test period as
the demo data (the prospective "what staff would have seen each week" story).
Shipped: `dashboard/build_dashboard.py` -> `dashboard/index.html` (94 KB, zero
dependencies - the one chart is hand-rolled SVG, so no 3.5 MB Plotly vendoring).
Weekly view with alert-budget selector (1/2/3/5/10 flags/wk, frozen thresholds),
flagged athletes with top-3 logistic contributions per flag ("vs own history"
features dominate, consistent with 03), risk bands (High/Elevated/Watch) instead of
percentages, retrospective outcome tags, and the "when not to trust this" panel on
the page. The builder refits the exact 04 winner and asserts it reproduces the saved
test AP/AUC before writing anything. Generated HTML is gitignored (embeds
athlete-day rows; data/README.md rule); screenshots committed in docs/img/ and
embedded in the README. Verified in browser, light and dark.

### Next session starts here

Resume wiring: Projects section + GitHub link, coordinated via the wiki
(`wiki/resume/JD - LMI - AI ML Engineer.md`, Application Status section). The
project is interview-ready as it stands; remaining polish is optional.

## Session 2 (2026-07-28 / 29): 01 -> 03, three corrections

Started from the committed scaffold, ended with a validated model within 0.035 AUC of
the published benchmark. Three things were caught and fixed along the way; all three
are portfolio material rather than embarrassments.

**Repo created:** https://github.com/auswallace/runner-injury-risk (public).

### 01_eda

Settled: 1.36% positive rate (583/42,766). 74 athletes, 43-1,791 rows each, 11 with
zero injuries. `Date` is a shared calendar index 0..2673 with staggered enrollment, so
a time-aware split is well defined. Zero NaN cells; a zero-km day is logged rest (the
perceived scores are filled in), and the real unlogged periods are gaps *between* rows.

**Correction 1 - slot ordering was backwards.** A first pass inferred from injury-row
load averages that the unsuffixed block was the newest day. Wrong. Replaced with a
structural proof: for consecutive-date rows a fixed calendar day's value slides from
`.6` toward the unsuffixed slot as it ages (row(t).k == row(t-1).(k+1)), 2,502,000
comparisons, zero mismatches. `.6` is the newest slot. Lesson recorded in the field
guide: averages suggest, structure proves.

### 02_features

First version reconstructed the per-athlete daily series from the overlapping windows
(exact: zero disagreement across overlaps, and rolling `km_7d` re-derived every source
row's own slot sum to 1e-13) and built 7d/28d load, coupled ACWR, 14-day rest and
week-over-week change. A validation check caught that negative day indices were real
pre-Date-0 training history, not zero padding.

### 03_model, first run: a 0.987 AUC that had to die

The v1 feature set produced validation AUC 0.987 / AP 0.49 on a 1.4%-prevalence
problem. The project's own red-flag rule (suspect anything above ~0.9 here) triggered
the hunt:

1. Not multi-day injury runs - all 583 positives are isolated onset days.
2. Not athlete memorization - athlete-grouped CV was equally inflated.
3. Autopsy: `km_28d`, `acwr`, `rest_days_14d`, `wow_km_change` were NaN for **100% of
   injury rows** vs 4-19% of healthy rows; one missing-indicator flag carried 93% of
   XGBoost's gain. The model had learned "window failed its coverage gate = injury."
4. Root cause in the data, not the code: every injury row's previous same-athlete row
   is >= 22 days back (median 22); 98.9% of healthy rows have one 1 day back.

**Correction 2.** Any feature window reaching past a row's own 7 days is systematically
empty for positives only, and *every* handling of that hole leaks - flags, imputation,
zero-fill, or dropping incomplete rows (which would delete all 583 positives). Austin's
call: document chronic load / ACWR as unbuildable on this dataset rather than ship
them. Later confirmed word-for-word in the paper's Methods ("for the healthy events,
we demanded that the athlete is fully fit 3 weeks before and 3 weeks after").

### Reading the source paper

Benchmark verified rather than guessed: day approach test AUC **0.724** (SD 0.01),
sens 0.584 / spec 0.741; week 0.678. Test set = the 10 most recently joined athletes,
so it maps to our athlete-grouped split. ROC only - no PR, no calibration table, no
alert budget.

**Correction 3 - window alignment.** The paper's window is the 7 days *before* the
event ("the day before the event is seen as day 0") and the target is whether the
*next* session brings injury. So `.6` is the day before the injury, not the injury
day. An intermediate version had dropped `.6` as a leakage precaution based on that
wrong assumption, which was costing real signal. Restored: `__w7` (paper-identical) is
the headline, `__w6` survives as a 2-day-lead sensitivity check.

**Per-athlete normalization adopted, with a deliberate deviation.** The paper
z-normalizes per athlete over that athlete's healthy events - including events *after*
the row being normalized, which is mildly transductive. We normalize against each
athlete's strictly earlier healthy rows (expanding, shifted, >= 20 required). Deviation
documented rather than copied. A NaN symmetry audit is now a standing gate on every
feature family (worst gap 0.075, in the healthy-missing-more direction; v1's poisoned
features were 0.81-0.96).

### 03_model, final

Per-athlete normalization roughly doubles AP - grouped CV AP 0.029 (raw) -> 0.064
(raw+z), AUC 0.658 -> 0.690. Winner under the pre-registered rule: **logistic
regression C=0.01 on `w7 raw+z`, both splits.** XGBoost lost every comparison -
complexity did not earn its keep. Class weights beat unweighted and 10:1
undersampling. Test sets never touched.

| | ours | paper |
|---|---|---|
| grouped AUC | 0.690 | 0.724 |
| grouped AP | 0.064 (4.5x chance) | not reported |
| time-aware AUC / AP | 0.634 / 0.032 | n/a |

### Decisions Austin made this session

- Sex subgroup is unsupportable (no per-athlete sex label ships; refused to infer it
  from training volume) -> replaced with tenure, volume-tier, and per-athlete-spread
  checks, documented as a limitation.
- Prospective framing leads; same-day kept only for benchmark comparison.
- Median-impute + missing-indicator flags over dropping incomplete rows.
- Symmetric windows after the leak; chronic load/ACWR documented as dead.
- 7-slot paper-aligned window as headline once the alignment was corrected.
- Per-athlete normalization in the leak-free form, accepting the deviation.

### Also produced

`docs/FIELD_GUIDE.md` - Austin's private study guide (plain-language framing, notebook
plan, every locked methodology decision as WHAT/WHY-HERE/NAIVE-FAILURE, an
"how to evaluate my AI assistant" checklist grounded in the three real corrections
above, and a 15-question self-test with answers). Quiz attempted through Q6; Q1-Q4
were misses or partials, Q5 (the ROC/precision arithmetic) was a clean unassisted hit.
Plan is to finish the build first, then re-quiz.

### Next session starts here

04_evaluation. Refit the winners on dev data, then evaluate **once** on the locked
test sets: PR/AP headline, ROC for paper comparison, calibration curve + Brier
(class-weighted probabilities will be miscalibrated - recalibrate on validation, never
on test), alert-budget table, subgroup checks, and the `__w6` lead-time sensitivity
run. Then the decision-support dashboard.
