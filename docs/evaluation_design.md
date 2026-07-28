# Evaluation design

Written BEFORE modeling, so the evaluation can't quietly bend to flatter the model.

## Benchmark

The source paper (Lovdal et al. 2021) reports AUC via bagged XGBoost on this exact
dataset (day approach and week approach variants). That published number is the
benchmark. Matching it with honest validation is success; beating it is a bonus;
understanding *why* results differ is the real deliverable.

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

Performance split by sex (only 27 of 74 athletes are women - small-n caveat gets
stated, not hidden) and by athlete tenure. If the model degrades for a subgroup,
that goes in the README, not under the rug.

## Assumptions log

Every modeling assumption gets a line in docs/assumptions_limitations.md at the
moment it is made, not retroactively.
