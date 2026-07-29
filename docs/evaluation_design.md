# Evaluation design

Written BEFORE modeling, so the evaluation can't quietly bend to flatter the model.

## Benchmark (verified from the paper, 2026-07-29)

Lövdal et al. 2021, bagged XGBoost (9 models on balanced 2048/2048 resamples, Platt
calibrated):

| approach | test AUC | validation AUC | sensitivity | specificity |
|---|---|---|---|---|
| day  | **0.724** (SD 0.01) | 0.729 | 0.584 | 0.741 |
| week | 0.678 (SD 0.01) | 0.783 | 0.504 | 0.746 |

Their test set is the data of the 10 most recently joined athletes, so their number
is closest to our **athlete-grouped** split, not our time-aware one. Matching it with
honest validation is success; beating it is a bonus; understanding *why* results
differ is the real deliverable.

**What the benchmark does not report, and we do.** The paper reports ROC only. At
this dataset's ~1.4% prevalence, its published day-approach operating point (58.4%
sensitivity, 74.1% specificity) implies roughly 3% precision - about 32 false alarms
per true flag. That is not a criticism of their modeling; it is the number an
operational program needs and ROC cannot show. Precision-recall, calibration and the
alert-budget view below exist to make it visible.

## Split strategy (both, reported separately)

1. **Time-aware:** train on early years, test on later years. Answers "would this
   have worked prospectively?" - the only question that matters operationally.
2. **Athlete-grouped:** no athlete appears in both train and test. Answers "does
   this generalize to runners it has never seen, or is it memorizing individuals?"

Plain random row splits are forbidden here: rows from the same athlete-week are
near-duplicates and leak.

## Metrics

- Precision-recall curve and average precision as the headline (injuries are rare;
  ROC curves flatter rare-event models)
- ROC AUC for comparability with the paper
- Calibration curve + Brier score: if the model says 10% risk, is it right ~10% of
  the time? A triage score that isn't calibrated is decoration
- Alert-budget view: at a realistic "flags per week the staff could act on"
  threshold, what do precision and recall look like?

## Subgroup checks

Sex-based subgroup analysis was planned but is not supportable: the cohort is
described as 27 women / 47 men in the source paper, but the released data does not map
sex to athlete IDs, and we do not attempt to infer it. Recorded as a limitation rather
than silently dropped. Feasible subgroup checks instead: (a) athlete tenure -
short-history vs long-history athletes; (b) training-volume tier - high-km vs low-km
athletes; (c) per-athlete performance spread - distribution of per-athlete metrics, to
detect a model that only works for a subset of runners while looking fine on average.

With 74 athletes, an average metric can hide a model that is useless for a third of
them; check (c) is the one an operational program would actually care about. If the
model degrades for a subgroup, that goes in the README, not under the rug.

## Assumptions log

Every modeling assumption gets a line in docs/assumptions_limitations.md at the
moment it is made, not retroactively.
