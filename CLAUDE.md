# Project brief: runner-injury-risk

Read this fully before doing anything. It is the complete context handoff from the
planning sessions that scaffolded this repo.

## What this is and why it exists

Austin (Senior Information Security Analyst, Federal Reserve Bank, 7 yrs of security
data analytics: Tableau, Python/scikit-learn, Databricks) is applying to an
**AI/ML Engineer (Data Scientist)** role at LMI supporting the U.S. Army's Holistic
Health & Fitness (H2F) program - injury-risk and readiness modeling inside a system
called H2FMS. This repo is his portfolio project: the public-data version of that
exact problem, built to demonstrate the JD's core loop - **build, validate,
operationalize** statistical/ML models - plus the things the JD explicitly names:
feature engineering, model evaluation, handling data quality issues, documenting
assumptions/limitations/bias, and integrating outputs into decision-support tools.

Deeper spec lives in Austin's Obsidian wiki: `wiki/projects/Runner Injury Risk Model
(H2F Portfolio Project).md`. The tailored resume and JD analysis are in
`wiki/resume/`.

**Labeling rule (hard):** independent project on public data. No claimed affiliation
with the Army, H2F, H2FMS, or LMI - anywhere, ever. No Army branding.

## The dataset (already in data/raw/, gzipped - pandas reads .csv.gz directly)

Groningen open replication data (Lovdal, den Hartigh & Azzopardi 2021, IJSPP,
DOI 10.34894/uwu9pv; data mirror pulled from GitHub). Two framings:

- `day_approach_maskedID_timeseries.csv.gz` - 42,766 rows x 73 cols
- `week_approach_maskedID_timeseries.csv.gz` - 42,798 rows x 72 cols
- 74 athletes (27 W / 47 M), 7 years of training logs, masked IDs
- **Injury rate ~1.36% (583 positive rows in day framing)** - rare-event problem
- Features: km by intensity zone, sessions, perceived exertion, strength training,
  rest days, etc.

**Structure of a row (proven in 01, alignment confirmed against the paper):** each
row is one athlete-day carrying a 7-day window of 10 metrics. Slot `.6` is the
**newest** slot and the unsuffixed block the oldest. All 7 slots are days *before*
the event - the paper's target is whether the *next* training session brings injury -
so `.6` is the day before the injury, NOT the injury day.

**Dataset artifact that constrains all feature design (found in 03, then confirmed in
the paper's Methods):** healthy events required the athlete to be fully fit 3 weeks
either side, so every injury row's previous same-athlete row is >= 22 days back while
98.9% of healthy rows have one 1 day back. Any feature window reaching past a row's
own 7 days is empty for injury rows only, and its missingness encodes the label.
**Chronic load, ACWR, 14-day rest and week-over-week change are unbuildable here** -
this was diagnosed by killing a 0.987-AUC model. Do not reintroduce them.

**Benchmark (verified 2026-07-29, no longer guess-forbidden - it is in
docs/evaluation_design.md):** bagged XGBoost day approach test AUC 0.724 (SD 0.01),
sens 0.584 / spec 0.741; week approach 0.678. Their test set is the 10 most recently
joined athletes, so it maps to our athlete-grouped split. They report ROC only.

## What we are measuring (decided - do not silently change)

Defined in `docs/evaluation_design.md`, written BEFORE modeling on purpose:

1. **Headline: precision-recall / average precision** (rare events make ROC flatter)
2. ROC AUC only for comparability with the paper
3. **Calibration curve + Brier score** - a triage score must mean what it says
4. **Alert-budget view** - at a realistic flags-per-week threshold, what are
   precision and recall? This is the operational framing an H2F-style program cares
   about
5. Subgroup checks with small-n caveats stated openly. Sex subgroup was found
   unsupportable (no per-athlete sex label ships with the data; we do not infer it)
   and was replaced with tenure, volume-tier, and per-athlete-spread checks - see
   docs/evaluation_design.md
6. Two split strategies, reported separately: time-aware (train early years / test
   late) and athlete-grouped (no athlete in both sides). Plain random row splits are
   forbidden - leakage.

## Method plan (order of attack)

1. `01_eda` - class balance, missingness (first open question: are missing days
   "rest" or "unlogged"? this decision shapes every feature), athlete heterogeneity.
   Every data quality finding goes into `docs/assumptions_limitations.md`.
2. `02_features` - ~~acute (7d) vs chronic (28d) load + ratio~~ SUPERSEDED: the
   chronic side of the plan died on the dataset artifact above. What shipped:
   within-window features on the 7 pre-event days + leak-free per-athlete
   normalization (the paper's key trick, minus its transductive shortcut).
3. `03_model` - regularized logistic regression with class weights FIRST (baseline),
   then XGBoost. Compare imbalance strategies (class weights vs resampling).
   Complexity must earn its keep. DONE - it didn't: logistic won everywhere.
4. `04_evaluation` - implement evaluation_design.md exactly.
5. Decision-support layer - weekly flagged-athletes view (top features per flag,
   confidence, "when not to trust this" panel). Tableau Public or self-contained
   HTML/Plotly. This is Austin's differentiator - do not skip it.

## Division of labor (hard rule - this is an integrity thing)

Every substantive analysis decision must be one **Austin makes and can defend cold
in an interview**: feature definitions, model choices, threshold selection,
interpretation of results. Claude's role: scaffolding, data plumbing, plotting
boilerplate, code review, rubber-ducking the decisions, catching leakage/stats
errors. When a modeling decision comes up, PRESENT OPTIONS AND TRADE-OFFS AND ASK -
do not just pick. If Austin can't explain a line of code, that line is a liability,
not an asset. The interview test for every merged change: "can Austin explain why
this exists and what breaks without it?"

On authorship: commits run under Austin's git identity (already configured). The
defensible interview posture is "I used AI tooling as an accelerator - the analysis
decisions are mine," which is only true if the rule above is followed. Do not
manufacture fake-looking "human" artifacts; make the human ownership real instead.

## Writing style (repo prose: README, docs, comments)

Plain hyphens, no em dashes. Concrete and quantified over adjectives. Notebook
markdown explains WHY, not just what. Honest about what didn't work - the README's
"what I tried that failed" material is a feature, not an embarrassment.

## Status + next actions

- [x] Scaffold committed (git initialized, branch `main`, Austin's identity)
- [x] Data in `data/raw/` (gitignored), verified: 74 athletes, ~1.36% positive rate
- [x] GitHub repo created and pushed: https://github.com/auswallace/runner-injury-risk
- [x] 01_eda: rest-vs-unlogged answered (zero-km = logged rest; gaps between rows
      are the true unlogged periods); Date is a shared calendar index; slot ordering
      proven structurally (.6 = newest slot = day before injury)
- [x] 02_features (v3): 12 features x 2 windows (`__w7` = paper-identical 7 slots,
      `__w6` = 2-day-lead sensitivity check) + per-athlete z-normalization (`__z`),
      leak-free variant: each row scored against that athlete's strictly EARLIER
      healthy rows only, >= 20 required. NaN symmetry audit is a standing gate on
      every new feature family (worst gap 0.075 vs 0.81-0.96 on v1's poisoned set)
- [x] 03_model (v3): winner = logistic C=0.01 on w7 raw+z, both splits, by the
      pre-registered rule (max val AP, ties to simpler). Grouped CV AUC 0.690 /
      AP 0.064 (chance 0.014); time-aware AUC 0.634 / AP 0.032 (chance 0.016).
      XGBoost lost every comparison. Splits: time-aware T1/T2 at Date quantiles
      .6/.8 with 7-day purge; grouped = injury-sorted every-4th athlete to test.
      Chosen configs + benchmark in data/processed/model_selection.json
- [x] 04_evaluation: one-shot test run DONE (numbers final, do not revisit). Test
      AP 0.030 both splits (1.7x chance TA / 2.6x GR); AUC 0.629 TA / 0.619 GR vs
      paper 0.724. Grouped CV->test drop (0.690->0.619) = per-athlete heterogeneity
      (per-athlete AUC 0.27-0.83, 2 of 15 below 0.5). Platt-calibrated Brier ties
      base rate: score is a ranking, not sharp probabilities - bands, not
      percentages. Alert budget: 2.9-4.3% precision at 1-5 flags/wk, recall 2-18%.
      w6 lead-time is ~free. Four pre-test decisions (train-only refit + val
      calibrator, Platt, frozen val thresholds, median subgroups) locked by Austin,
      logged in assumptions_limitations.md; results in
      data/processed/evaluation_results.json; blind spots filled into "When NOT to
      trust the score"
- [ ] **Dashboard layer - NEXT** (decision-support view - Austin's differentiator,
      do not skip). Reads evaluation_results.json + per-row scores; must carry the
      calibration caveat (bands not percentages) and per-athlete-spread caveat on
      its face
- [ ] Then: Projects section + GitHub link on the resume (coordinate via the wiki,
      see `wiki/resume/JD — LMI — AI ML Engineer.md` Application Status section)

Meta-reminder for Austin, not the code: the LMI application ships BEFORE this
project is finished. Half-done with a good README is already interview material.
