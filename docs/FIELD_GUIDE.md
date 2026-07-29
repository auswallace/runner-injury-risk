# Field guide (for Austin, not the public)

Purpose: you defend every choice in this repo cold, in an interview, with no AI in
the room. Everything here ties to this repo's actual files and numbers. If a claim
in here surprises you, open the notebook it cites and re-derive it before moving on.

---

## 1. The question, in plain language

We watch how much and how hard 74 competitive runners train, day by day, for 7
years. We ask: in the days before a runner gets injured, does the training pattern
look different enough that a computer could have raised a hand in advance? If yes,
a coach with limited time could check on the few most at-risk runners each week
instead of guessing.

Why an operational program cares: an Army injury-prevention team (or any team
sports-medicine staff) cannot examine everyone every week. What they need is
triage - a short, trustworthy list of who to look at first, with an honest label
on how often the list will be wrong. That is exactly what this project builds and
measures. The rare-event math, the messy longitudinal logs, and the "can staff act
on this?" framing are the same whether the population is elite Dutch runners or
soldiers in a readiness program.

---

## 2. The plan, notebook by notebook

### 01_eda - data quality pass (DONE)

Question it answers: what is actually in this data, and which properties change
every downstream choice?

What it found (each proven in the notebook, not assumed):

- 42,766 rows, 583 positives = 1.36% injury rate. 1 positive per ~73 rows.
- 74 athletes, wildly heterogeneous: 43 to 1,791 rows each (median 426), and 11 of
  74 have zero injury rows.
- `Date` is a shared calendar index 0..2673 (~7.3 years), ~16 athletes active per
  date. So "train early / test late" is well defined.
- Missingness: zero NaN cells. A zero-km day is a logged rest day (the perceived
  scores are still filled in). The true "unlogged" periods are gaps BETWEEN rows -
  usually 1-day steps, occasionally weeks to months.
- Window ordering: each row is a 7-day sliding window. Slot `.6` is the most recent
  day (the injury day on positive rows); the unsuffixed block is the oldest (t-6).
  Proven by 2,502,000 slide comparisons with zero mismatches - NOT by eyeballing
  averages (the eyeball version got it backwards; see section 4).

Done means: every finding has a proof cell in the notebook and a line in
`docs/assumptions_limitations.md`.

### 02_features - feature engineering (DONE)

Question it answers: can we turn an already-windowed table into the features the
sports-science framing wants (7-day acute vs 28-day chronic load), without leaking
the future or fabricating data?

Key move: one row only sees 7 days, but consecutive rows overlap 6 of 7 days, so
the per-athlete daily series is recoverable EXACTLY. Validation: zero disagreement
across all overlapping copies (49,162 daily records from 299,362 slot records),
and the rebuilt rolling `km_7d` reproduces every source row's own 7-slot sum to
1e-13.

Output (v2, after the leak discovery - see the 03 post-mortem): 12 features x 2
framings in `data/processed/features_day.csv.gz` (gitignored - athlete-day tables
could reconstruct an individual's log, which `data/README.md` forbids committing):

- load: `km_sum`, `sessions`, `km_mod`, `km_hi`
- intensity mix: `pct_mod`, `pct_hi` (NaN on zero-running windows - value-based,
  symmetric: 3.4% of positives vs ~10% of negatives)
- recovery: `rest_days`, `recovery_avg`
- other load: `strength_n`, `alt_hours`
- subjective: `exertion_avg`, `success_avg`

Two framings per feature, computed from each row's OWN window only: `__same`
(days t..t-6, the paper's framing, includes the injury day) and `__pros`
(days t-1..t-6 - 6 days, identical for every row, nothing from the prediction day).

DEAD, and you must be able to say why cold: `km_28d`, `acwr`, `rest_days_14d`,
`wow_km_change`. The publishers removed all rows for >= 22 days before every
injury, so any window reaching past the row's own 7 days is empty for positives
only - every possible handling of that emptiness encodes the label (v1: 100% NaN
on positives, one missing flag = 93% of XGBoost gain, 0.987 fake AUC; and
complete-case would have deleted all 583 positives).

Done means: reconstruction proven exact, the buffer artifact demonstrated, both
framings emitted symmetric, the chronic-load failure documented loudly.

### 03_model - modeling (DONE, twice - v1 died of leakage)

Question it answers: how much signal is there, and does model complexity earn its
keep?

The arc you must own: v1 (with chronic-load features) posted validation AUC 0.987 /
AP 0.49 - our own checklist said "suspect leakage above 0.9," and the hunt found
the publisher's >= 22-day pre-injury row buffer. Features rebuilt, notebook rerun.

v2 results (leak-free, prospective 6-day features):

- Splits: time-aware with 7-day purge gaps (train 25,652 rows / 300 pos;
  val 8,481 / 132; test locked 8,462 / 149) and athlete-grouped (56 dev / 18 test
  athletes, injury-sorted every-4th assignment, GroupKFold(4) selection)
- **Logistic C=0.01 won BOTH splits** under the pre-registered rule (max val AP,
  ties to simpler): time-aware val AP 0.021 (chance 0.016), AUC 0.60; grouped mean
  AP 0.029 (chance 0.014). XGBoost lost everywhere - complexity did not earn its
  keep on 6-day windows
- Imbalance comparison: class weights ~= unweighted ~= undersampling on AP
  (ranking metrics barely move; weighting matters for calibration later)
- The honest headline: prospective signal is MODEST, ~1.4x chance AP. Say that
  plainly and then explain why the 0.987 number was worthless - that contrast IS
  the portfolio

Done means (met): a baseline nobody can accuse of leakage, a complexity verdict,
winners + rule saved to `data/processed/model_selection.json`, assumptions logged.

### 04_evaluation - implement docs/evaluation_design.md exactly (PENDING)

Question it answers: would this have worked, for whom, and can the score be
trusted at face value?

Deliverables: both splits reported separately, PR/AP headline + ROC AUC for the
paper comparison, calibration curve + Brier, the alert-budget table, subgroup
checks (tenure, volume tier, per-athlete spread). Before running it: look up the
paper's actual reported AUC numbers - the brief forbids guessing them.

Done means: every item in evaluation_design.md has a cell, including the ones
that make the model look worse.

### 5th piece - decision-support layer (PENDING, your differentiator)

Weekly flagged-athletes view: who is flagged, top contributing features per flag,
confidence, and a "when not to trust this" panel fed by the calibration and
subgroup results. Self-contained HTML/Plotly or Tableau Public.

---

## 3. Locked methodology decisions - WHAT / WHY HERE / NAIVE FAILURE

### 3.1 Class imbalance (~1.36% positives)

- WHAT: only 583 of 42,766 rows are injury days. A model that says "never injured"
  is 98.6% accurate and 100% useless.
- WHY HERE: this is the defining property of the dataset. It drives the metric
  choice (3.4), the loss weighting plan in 03 (class weights vs resampling), and
  the alert-budget framing (3.7).
- NAIVE FAILURE: train an unweighted classifier on raw rows and report accuracy.
  You get a degenerate always-negative model with a great-sounding number. Second-
  order trap: resampling the TEST set - oversampling positives in evaluation
  inflates precision, because the real world stays 1.36%. Resampling is a training
  trick only; the test set must keep the true prevalence.

### 3.2 Data leakage - the specific vectors in THIS repo

- WHAT: leakage is any path by which information unavailable at prediction time
  reaches the model. Four live vectors here:
  1. Row overlap: adjacent rows share 6 of their 7 days, so a random split puts
     near-copies of the same window on both sides.
  2. Athlete identity: heterogeneity is huge (11 athletes have zero injuries), so
     a model can score by memorizing WHO instead of learning WHAT precedes injury.
  3. Time: training on 2018 to predict 2015 answers no operational question.
  4. Same-day slice: slot `.6` is the injury day's own training. A "prediction"
     using it is partly describing the event, not forecasting it.
- WHY HERE: every one of these is present simultaneously, which is why the split
  design (3.3) and the two framings (02) exist.
- NAIVE FAILURE: random row split + same-day features gives a spectacular AUC that
  evaporates the day someone deploys it. This is the single most common way
  published time-series models fail to replicate.

### 3.3 Time-aware AND athlete-grouped splits, never random

- WHAT: (a) train on early calendar, test on late calendar; (b) train on some
  athletes, test on entirely unseen athletes. Reported separately, never averaged.
- WHY HERE: they answer different operational questions. Time-aware: "would this
  have worked prospectively?" - the deployment question. Athlete-grouped: "does it
  transfer to a runner it has never seen, or is it memorizing individuals?" - the
  new-cohort question. 01 proved `Date` is a shared calendar, so (a) is well
  defined. Caveats you must volunteer: the early/late athlete mixes differ
  (athletes enroll and drop out), and grouped test folds can draw injury-poor
  athletes (11/74 have zero positives), so folds must report positive counts.
- NAIVE FAILURE: random row split. Via the 6/7-day overlap the model has already
  seen a near-copy of most test windows, and via athlete memorization it scores on
  identity. Numbers go up; meaning goes to zero.

### 3.4 Precision-recall over ROC as headline; why ROC flatters rare events

- WHAT: headline is the PR curve and average precision; ROC AUC is reported only
  to compare with Lovdal et al.
- WHY HERE - with this dataset's numbers: ROC's x-axis is false positive rate =
  FP / 42,183 negatives. A "tiny" 5% FPR is ~2,100 false alarms. If the model
  catches 60% of the 583 positives (350) at that FPR, the ROC point (0.05, 0.60)
  looks strong - but precision is 350/(350+2,100) = 14%: six of every seven flags
  are false. The huge negative denominator hides false alarms; precision's
  denominator (flags raised) exposes them. Precision is what the coach experiences.
- NAIVE FAILURE: lead with ROC AUC. You can report 0.9+ while the actual alert
  stream is 90% noise, and staff learn to ignore the tool - the alert-fatigue
  failure you know from security operations.

### 3.5 Average precision (AP)

- WHAT: area under the PR curve - average of precision across recall levels.
- WHY HERE: one threshold-free summary whose floor is honest: a random scorer gets
  AP ~= prevalence = 0.014 here (vs ROC's flattering 0.5 floor). AP = 0.10 reads
  correctly as "7x better than chance, still noisy," which is the true state of
  injury prediction. Threshold-fixed metrics like F1@0.5 are meaningless here
  because 0.5 is an arbitrary cut on a score whose base rate is 0.0136.
- NAIVE FAILURE: report accuracy or F1 at the default threshold - you measure the
  threshold convention, not the model.

### 3.6 Calibration curve + Brier score

- WHAT: calibration checks whether "10% risk" events happen ~10% of the time
  (reliability diagram); Brier score is mean squared error of the predicted
  probability against the 0/1 outcome.
- WHY HERE: the deliverable is a triage SCORE, not a binary label. Staff will
  rank and budget by it; an uncalibrated score is decoration. Class-weighting and
  resampling in 03 deliberately distort base rates, so raw model outputs will NOT
  be calibrated - 04 must check and likely recalibrate. Brier caveat you must
  volunteer: with 1.36% prevalence, always-predict-0.0136 gets Brier ~= 0.0134, so
  a small Brier alone proves nothing - it needs the curve and a skill comparison
  next to it.
- NAIVE FAILURE: ship `predict_proba` from a class-weighted model. It might say
  "40% risk" for athletes whose true risk is 4%. First week of use, a coach
  notices the numbers are fantasy and the tool is dead.

### 3.7 Alert-budget view

- WHAT: fix the operational capacity first - "staff can review N athletes per
  week" - then read precision and recall AT that flag volume, instead of picking
  an abstract probability threshold.
- WHY HERE: it converts model quality into the only currency an operational
  program spends: staff attention. "At 5 flags/week you catch X% of injuries and
  Y% of flags are true" is a sentence a program manager can act on; "AUC 0.78" is
  not. This is the project's bridge from model to decision-support tool, and the
  framing the H2F-shaped role actually cares about.
- NAIVE FAILURE: threshold at 0.5 (or max-F1) and report the confusion matrix.
  The implied workload is whatever it happens to be - possibly 300 flags/week
  nobody can process, possibly 0 - and the tool answers a question nobody asked.

### 3.8 Acute:chronic workload ratio (ACWR)

- WHAT: last 7 days' load divided by the 28-day load expressed per week - coupled
  form: `km_7d / (km_28d / 4)`. Sports-science motivation (Hulin et al. 2016,
  Gabbett 2016): spikes above habitual load precede injury.
- WHY HERE: adopted as DOMAIN GUIDANCE for feature design, explicitly not
  invented here and explicitly not treated as an injury law - Impellizzeri et al.
  2020 critique the causal reading, and EWMA variants (Williams et al. 2017)
  exist. It is a candidate feature; the model decides if it earns its keep.
  Sub-decisions you own: coupled (not uncoupled) form, and coverage gates so a
  post-gap "7-day window" spanning months of real time returns NaN instead of a
  fake ratio. UPDATE after 03's leak hunt: ACWR turned out to be uncomputable
  without artifact on this dataset - every injury row has NaN chronic load because
  the publishers removed rows for >= 22 days before each injury. See the leak
  post-mortem in 03. (An earlier draft of this section claimed 02's ACWR-bin plot
  showed a U-shape; the plot actually showed zero injuries in every bin - all
  positives had NaN ACWR - which was itself the artifact announcing itself.)
- NAIVE FAILURE: zero-fill the gaps and compute ACWR anyway. A runner returning
  from a 2-month unlogged layoff gets chronic ~= 0 and an enormous ratio - the
  model then "discovers" that returning from a gap is risky, which may be real
  but is measured garbage. The gates make that honest (NaN), at a stated cost
  (~19-20% of rows).

### 3.9 Sex subgroup check - replaced, and why

- WHAT: the plan called for performance split by sex. The released tables carry
  only masked `Athlete ID` - the 27 W / 47 M figure is a cohort description in
  the paper, never mapped per athlete. Replaced with: tenure split, volume-tier
  split, and per-athlete performance spread. Recorded as a limitation in
  `docs/evaluation_design.md` and `docs/assumptions_limitations.md`.
- WHY HERE: you cannot check what the data does not encode, and inferring sex
  from training volume would be both methodologically indefensible and ethically
  wrong - so we refused. The per-athlete-spread check is the sleeper: with 74
  athletes, one averaged AUC can hide a model that is useless for a third of
  them, and that distribution is what an operational program actually needs.
- NAIVE FAILURE: either silently drop the fairness check (reviewer asks "where
  did it go?" - integrity hit), or proxy sex from volume (indefensible). The
  defensible story is the one that happened: planned it, found the data could
  not support it, refused to proxy, documented it, substituted checks the data
  CAN support. That sequence is a direct answer to the JD's "assess limitations,
  assumptions, and potential bias."

---

## 4. How to evaluate my AI assistant

Two real errors it made in THIS repo - both caught by verification, not by trust:

- Window-ordering reversal: 01_eda originally claimed the unsuffixed block was the
  most recent day, inferred from a univariate average. Backwards. The structural
  slide check (2.5M comparisons) proved `.6` is most recent. Lesson: averages
  suggest, structure proves.
- Padding deletion: 02 originally dropped negative day indices as "all-zero
  padding." All 206 such records carry real training data (athlete 0, day -1:
  16.4 km). The anchor check failed by exactly 16.4 km and exposed it. Lesson:
  build checks that reconcile derived data against the source, then trust the
  check, not the narrative.

### Leakage red flags - reject or interrogate on sight

- [ ] Any split without explicit grouping/time logic (`train_test_split` with no
      `groups`, no date cutoff)
- [ ] Scaler/imputer/encoder fit before the split, or on train+test together
- [ ] Features whose window can touch day t when the label is day t, in anything
      labeled prospective (`__pros` must end at t-1 - check the shift)
- [ ] Resampling (SMOTE/over/under) applied before splitting, or to the test set
- [ ] Rolling windows computed across athlete boundaries (one athlete's history
      bleeding into another's features)
- [ ] Threshold, calibration, or feature selection tuned on the test set

### Metric red flags

- [ ] AUC > ~0.9 on this problem - the published benchmark is nowhere near
      perfect; suspect leakage before celebrating
- [ ] Test metrics better than train metrics
- [ ] Precision computed on a resampled/balanced test set (real prevalence is
      1.36% - precision at 50/50 is fiction)
- [ ] Accuracy mentioned at all; F1 at default 0.5 threshold
- [ ] Metrics averaged over grouped folds without per-fold positive counts
      (11/74 athletes have zero injuries - a fold can be near-empty of positives)
- [ ] A single headline number with no calibration check attached

### Questions to ask before merging any change

1. "Prove the ordering/join/window is what you claim - show me the check cell and
   its output." (This exact demand caught both real errors above.)
2. "What information does this feature have access to, as of the morning of the
   prediction day?"
3. "If this number is this good, what leak would produce it? Rule that out first."
4. "What breaks if I delete this line/feature/gate?" (If neither of us can
   answer, it does not merge - CLAUDE.md interview test.)
5. "Which decision in this diff is mine to make, and did you make it for me?"
   (Framing choices, thresholds, feature definitions are yours; plumbing is not.)
6. "Rerun the notebook top to bottom clean, then show the error count."

---

## 5. Self-test - 15 questions, hardest last

Attempt each cold before opening the answers. Anything you miss is your next
study block.

1. What does one row of the day-approach table represent, and what exactly is the
   label saying?
2. The injury rate is 1.36%. Why is accuracy meaningless here, and what accuracy
   does an always-healthy predictor get?
3. Why are plain random row splits forbidden in this repo, specifically -
   what two mechanisms leak?
4. The evaluation runs BOTH a time-aware split and an athlete-grouped split.
   What different question does each answer, and why report them separately?
5. Walk through why ROC flatters this problem, with the actual numbers: what
   does 5% FPR mean in false alarms, and what precision does 60% recall at 5%
   FPR imply?
6. What is average precision, what is its chance-level floor on this data, and
   why is that floor more honest than ROC's?
7. What does it mean for the risk score to be calibrated, why will the raw 03
   model outputs probably NOT be, and why is a small Brier score alone not
   evidence of skill here?
8. Explain the alert-budget view to a non-technical program manager in two
   sentences, and say why it beats a 0.5 threshold.
9. Define the ACWR as computed in 02 (exact formula), name its intellectual
   source, and give the strongest published critique of using it causally.
10. The day table already ships as 7-day windows. How did 02 recover a daily
    series from it, and what two validations make you say "exact" out loud?
11. A zero-km day and an absent day are different things in this data. How do
    you know, what does each one mean, and what do the coverage gates do about
    the absent ones - why not just fill zeros?
12. Describe the window-ordering mistake: what was claimed first, what proved it
    wrong, and what does the corrected ordering reveal about when the load bump
    happens relative to injury?
13. Same-day vs prospective framing: what exactly leaks in the `.6` slice, why
    keep the same-day variant at all, and which one should lead the README?
14. An interviewer says: "Your fairness section is weaker than planned - you
    dropped the sex analysis." Give the full defense, including why you refused
    to infer sex and what replaced the check.
15. Your athlete-grouped results come back much worse than your time-aware
    results. What are the candidate explanations, how would you distinguish
    them, and what would each mean for deploying this on a NEW population -
    which is the deployment that actually matters for the role this project
    targets?

<details>
<summary>Answers - open only after attempting all 15</summary>

1. One row = one athlete on one calendar day (`Date` t), carrying a 7-day window
   of 10 training metrics: slot `.6` = day t (most recent), unsuffixed = day t-6.
   The label `injury` says whether THIS athlete got injured on day t. It is a
   day-level event label, not "injured sometime soon."

2. Predicting "never injured" is right on 42,183 of 42,766 rows = 98.6% accuracy
   with zero recall. Accuracy rewards the majority class; every metric here must
   be positive-class-aware (PR, AP, recall at a flag budget).

3. (a) Overlap: adjacent rows share 6 of 7 days, so random splitting puts
   near-copies of the same window in train and test - the model is graded on
   material it has effectively seen. (b) Athlete identity: with extreme
   per-athlete heterogeneity (43-1,791 rows, 11 athletes injury-free), rows from
   the same athlete on both sides let the model score by memorizing who, not
   what.

4. Time-aware = "would this have worked prospectively?" - the deployment
   question; train on the past, test strictly on the future. Athlete-grouped =
   "does it transfer to unseen runners?" - the new-cohort question. Separately,
   because they fail differently: averaging them hides which failure you have.
   Caveats: athlete mix differs across eras; grouped folds need per-fold
   positive counts because some folds draw injury-poor athletes.

5. FPR = FP/42,183 negatives, so 5% FPR = ~2,100 false alarms. Catching 60% of
   583 positives = 350 true alarms. Precision = 350/(350+2,100) ~= 14% - six of
   seven flags false - yet (0.05, 0.60) plots as a strong ROC point. The massive
   negative denominator absorbs false alarms; PR puts them in the denominator
   the operator experiences.

6. AP = area under the precision-recall curve (precision averaged over recall
   levels). Chance level ~= prevalence = 0.0136 here, so AP is read as a
   multiple of base rate (AP 0.10 ~= 7x chance). ROC AUC's floor is 0.5
   regardless of prevalence, which lets a barely-better-than-chance rare-event
   model look respectable.

7. Calibrated = among rows scored ~10%, ~10% actually have injuries (checked
   with a reliability diagram). 03's class weights / resampling deliberately
   distort the training base rate, so raw probabilities come out inflated -
   expect to recalibrate (e.g. isotonic/Platt on a validation fold). Brier: the
   trivial always-0.0136 predictor scores ~0.0134, so a small Brier is nearly
   free - it only means something next to the curve and against that trivial
   baseline.

8. "Tell me how many athletes your staff can actually check each week - say
   five. I will tell you what fraction of those five flags tend to be real
   concerns, and what share of injuries we catch at that pace." It beats 0.5
   because it starts from the resource that actually constrains the program -
   staff attention - instead of an arbitrary score cutoff that implies some
   unexamined, possibly absurd workload.

9. `acwr = km_7d / (km_28d / 4)` - last 7 days' km over the 28-day km expressed
   per week, coupled form (the acute week is inside the chronic window).
   Source: Hulin et al. 2016 / Gabbett 2016 - adopted as domain guidance.
   Strongest critique: Impellizzeri et al. 2020 - the ratio's causal "sweet
   spot" reading is confounded and study designs behind it are weak; also EWMA
   weighting (Williams et al. 2017) arguably represents recency better.
   CRITICAL UPDATE you must volunteer: on THIS dataset ACWR proved uncomputable
   without artifact - the publishers removed rows for >= 22 days before each
   injury, so chronic windows are empty for positives only and the missingness
   itself leaks the label (v1 posted a fake 0.987 AUC off one missing flag).
   The feature was removed and the failure documented - a better interview
   answer than any ACWR coefficient would have been.

10. Consecutive rows overlap 6/7 days; each calendar day appears in up to 7 rows
    at successive slots. Melt all slots to (athlete, day, metric), then:
    (a) every overlapping copy of the same (athlete, day) agrees exactly - max
    disagreement 0.0 across 299,362 slot records collapsing to 49,162 days;
    (b) anchor check - rolling 7-day `km_7d` recomputed from the reconstruction
    reproduces every one of the 42,766 source rows' own slot sums to ~1e-13.
    Exact, not estimated.

11. Zero-km rows still carry perceived-exertion/recovery scores, so the athlete
    was logged and rested - a real 0. Absent days are calendar days covered by
    no row's window - unlogged, unknown. Gates: a 7d window needs >=5 logged
    days (14d >=10, 28d >=20) or the feature is NaN. Zero-filling would turn
    unknown into "rested," manufacturing spurious load spikes after layoffs -
    especially poisonous for ACWR (chronic ~= 0 makes the ratio explode).

12. First claim: unsuffixed block = most recent day, inferred from injury-row
    load averages. Proof of the reverse: for consecutive-date rows,
    row(t) slot k equals row(t-1) slot k+1 - a day's value slides from `.6`
    toward unsuffixed as it ages - 2,502,000 comparisons, zero mismatches. So
    `.6` is the injury day. Corrected reading: the load elevation (~8.05 vs
    ~7.02 km baseline) sits at t-6, ~6 days BEFORE injury, and the injury day
    itself is near baseline - a spike-then-break shape, noted as a look, not a
    result.

13. `.6` is the injury day's own training - using it means "predicting" an event
    partly from the event's own day (a runner mid-injury logs differently), so
    the prospective framing shifts every window to end at t-1. Keep same-day
    anyway because the paper's benchmark is same-day - comparability requires
    it. Lead with prospective (it is the only framing a deployed triage tool
    could ever have) and label the same-day comparison honestly. Final call is
    yours - this is the flagged open decision for 03.

14. "The data cannot support it: the released tables carry only masked athlete
    IDs; 27 W / 47 M exists only as a cohort description in the paper, never
    per athlete. I refused to infer sex from training patterns - that would be
    methodologically indefensible and ethically wrong. I documented the gap as
    a limitation instead of silently dropping the check, and substituted
    subgroup analyses the data does support: tenure, volume tier, and
    per-athlete performance spread. The spread check is arguably stronger for
    operations: with 74 athletes, one average can hide a model that fails a
    third of them." Planned - found unsupportable - refused to proxy -
    documented - substituted. That sequence IS the bias-assessment competency.

15. Candidates: (a) the time-aware number is inflated by athlete memorization -
    same athletes on both sides of the date cutoff, so it partly measures
    identity recall; (b) genuine population shift - athletes differ enough that
    cross-athlete transfer is intrinsically hard; (c) grouped-fold artifact -
    test folds drew injury-poor athletes (11/74 have zero positives) making
    metrics unstable. Distinguish: for (c), check per-fold positive counts and
    rerun with different fold seeds; for (a), fit on athlete-demeaned /
    within-athlete features, or test whether adding athlete-identity proxies
    closes the gap - if time-aware drops toward grouped, memorization was the
    inflation; for (b), inspect per-athlete score spread among unseen athletes.
    Meaning for deployment: a new population (a fresh team, a different cohort)
    experiences the GROUPED number, not the time-aware one - so if grouped is
    much worse, the honest claim is "this tool personalizes to a monitored
    cohort over time; cold-start performance on strangers is X." For an
    H2F-shaped program onboarding new soldiers constantly, the grouped number
    is the one to put in front of the decision-maker.

</details>
