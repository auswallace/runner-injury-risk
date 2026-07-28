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

The companion paper's reported results (bagged XGBoost, ROC analysis) are the
**public benchmark** - look them up in the paper before evaluation, don't guess them.

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
2. `02_features` - acute (7d) vs chronic (28d) load + ratio, intensity distribution,
   recovery patterns, week-over-week change. Backward-looking windows only, computed
   per athlete. Cite sports-science motivation (acute:chronic workload ratio) as
   adopted domain guidance, not invention.
3. `03_model` - regularized logistic regression with class weights FIRST (baseline),
   then XGBoost. Compare imbalance strategies (class weights vs resampling).
   Complexity must earn its keep.
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
- [x] 01_eda: rest-vs-unlogged answered (zero-km = logged rest; gaps between rows are
      the true unlogged periods). Window ordering corrected and proven: .6 = most
      recent day / injury day, unsuffixed = t-6
- [x] 02_features: daily series reconstructed exactly from overlapping windows;
      16 features x 2 framings (same-day / prospective) in data/processed/
- [ ] 03_model, 04_evaluation per method plan above. Open decisions for Austin:
      headline framing (prospective vs same-day), ACWR form + coverage-gate
      sensitivity
- [ ] Dashboard layer
- [ ] Then: Projects section + GitHub link on the resume (coordinate via the wiki,
      see `wiki/resume/JD — LMI — AI ML Engineer.md` Application Status section)

Meta-reminder for Austin, not the code: the LMI application ships BEFORE this
project is finished. Half-done with a good README is already interview material.
