# Assumptions & limitations

Living document - add a line the moment an assumption is made. Each entry: what we
assumed, why, and what breaks if it's wrong.

## Data assumptions

- [x] **Missing days resolved (01_eda).** There are zero NaN cells in either table, so
      there is no cell-level imputation problem. A zero-km day is a *logged rest day*,
      not an absent day: on days where every running metric is 0, a perceived-exertion
      score is still recorded. The real "unlogged" gaps live *between rows* - most
      consecutive rows step 1 calendar day, but a fraction jump by weeks to months
      (off-season / layoff). Consequence: a rolling window is 7 *logged* days, which
      after a gap can span a long real interval. Carry this into feature construction.
- [x] **`Date` is a shared calendar index (01_eda),** integer 0..2673 (~7.3 yr), with
      ~16 athletes sharing each value (max 35) and staggered enrollment. So the
      time-aware split on a global `Date` cutoff is well defined. Caveat: the athlete
      mix differs between early and late periods (athletes drop in/out), so a late-period
      test set is not the same population as the early-period train set - a second reason
      to also run the athlete-grouped split. No real-calendar anchor exists; only the
      ordering is used, nothing is invented.
- [ ] Injury labels mark true injury onset dates, not report dates (not yet verifiable
      from these files; carried as an open assumption).
- [ ] Masked athlete IDs are stable across the full period (assumed; consistent with the
      per-athlete contiguous calendar spans seen in 01_eda).

## Open decisions (surfaced in 01_eda, for Austin to settle)

- ~~**Injury-day slice in the window.**~~ *Settled by Austin before 03:* prospective
  framing (windows end at t-1) leads everywhere; the same-day framing (includes the
  `.6` injury-day slice) is run once in 04, clearly labeled, solely for comparison
  against the paper's benchmark. (Background, proven in 01: `.6` = most recent day =
  injury day; unsuffixed = t-6.)
- ~~**Sex subgroup source.**~~ *Settled:* no sex label ships with the data and we will
  not infer it; sex subgroup replaced by tenure / volume-tier / per-athlete-spread
  checks and recorded as a limitation. See Known limitations above and
  `evaluation_design.md`. (Open confirmation for Austin: verify the Dataverse original
  ships no athlete-attributes/codebook file the GitHub mirror dropped - the sandbox
  can't reach Dataverse.)

## Dataset-construction artifact (discovered hunting 03 v1's 0.987 AUC)

- **The publishers removed all rows for >= 22 days before every injury** (every
  injury row's previous same-athlete row is >= 22 days back, median 22; 98.9% of
  healthy rows have one 1 day back). Consequence: any feature window reaching past
  the row's own 7 days is systematically empty for positives only, and EVERY
  handling of that emptiness (NaN flags, imputation, zero-fill, row-dropping - the
  last would delete all 583 positives) encodes the label. v1's long-window features
  (km_28d, acwr, rest_days_14d, wow_km_change) were NaN for 100% of injury rows vs
  4-19% of healthy rows; one missing-indicator flag carried 93% of XGBoost's gain
  and drove validation AUC to 0.987. **Chronic load and ACWR are therefore
  unbuildable without artifact on this dataset, in any framing.** Features rebuilt
  (Austin's decision): each row's own window only - prospective = days t-1..t-6
  (6 days, symmetric for every row), same-day = t..t-6 (7 days, paper framing).
  The v1 entries below are kept as a record; git history preserves v1 in full.

## Feature-engineering decisions (02_features v1 - superseded, see artifact note above)

- **Daily series reconstructed from overlapping windows, validated twice.** The
  day-approach table's 7-day windows overlap, so the per-athlete daily series is
  recoverable exactly: zero disagreement across all overlapping copies, and rolling
  `km_7d` re-derives every source row's own slot sum to 1e-13. Not an estimate.
- **Negative `day` values are real history, not padding.** First labeled rows' windows
  reach before global Date 0 and carry genuine training data (all 206 such records are
  nonzero). Kept as observed days; labels only attach at Date >= 0. (A first draft
  dropped them as padding; the anchor check caught it.)
- **Unlogged gap days stay NaN; windows are coverage-gated, never zero-filled.**
  Zero-filling would fabricate rest. Gates: 7d window needs >=5 logged days, 14d >=10,
  28d >=20, else feature is NaN. Thresholds are judgment calls; sensitivity is cheap
  to test in 03. Cost: ~19-20% of rows lose ACWR, ~14-15% lose km_28d.
- **Coupled ACWR** (`km_7d / (km_28d/4)`, acute week inside chronic) per Hulin et al.
  2016 / Gabbett 2016, adopted as domain guidance with critiques noted (Impellizzeri
  et al. 2020; EWMA variant Williams et al. 2017). Treated as a candidate feature,
  not an injury law.
- **Both framings produced:** same-day (`__same`, window ends at t, paper-comparable)
  and prospective (`__pros`, window ends at t-1, what a triage tool would have).
  Which one 03 leads with is an open decision for Austin.
- Processed tables live in `data/processed/` and are **gitignored**: athlete-day level
  tables could reconstruct an individual's log, which data/README.md forbids
  committing.

## Modeling assumptions

- **Prospective framing leads (decided by Austin before 03).** Headline models use
  only `__pros` features (windows end the day before the prediction day) - the only
  information a deployed triage tool would have. The same-day framing is run once,
  clearly labeled, solely for comparison against the paper's benchmark.
- **Missing gated features: median-impute + missing-indicator flags (decided by
  Austin before 03).** Imputer fit on training data only, inside the pipeline. The
  indicator flags are kept as features deliberately: "this athlete's window failed
  the coverage gate" usually means a recent unlogged gap (layoff/off-season), which
  is plausibly informative about risk. Trade-off owned: imputed values are
  fabricated medians; the alternative (dropping ~20% of rows) would systematically
  remove post-gap rows - likely the riskiest moments. *Post-artifact note: with v2
  features only the value-based pct features carry NaN (zero-running windows,
  symmetric: 3.4% of positives vs ~10% of negatives), so this decision now has far
  smaller surface.*
- **Shared preprocessing for both model families (03).** XGBoost's native NaN
  handling is deliberately not used; both families get the same imputed matrix so
  the comparison isolates the model, not the plumbing.
- **Small grids on purpose (03).** ~300 train positives; a large hyperparameter
  search would mostly fit validation noise. Logistic C in {0.01, 0.1, 1}; XGBoost
  depth {2,3,4} x lr {0.05, 0.1}.
- **Pre-registered selection rule (03, stated before results):** winner = highest
  validation AP; ties within 0.005 go to the simpler model. Outcome: logistic
  C=0.01 won BOTH splits (time-aware val AP 0.021 vs chance 0.016, AUC 0.60;
  grouped mean AP 0.029 vs chance 0.014). XGBoost lost everywhere - complexity did
  not earn its keep on 6-day windows.
- **Honest signal statement:** with leak-free prospective features the signal is
  modest (~1.4x chance AP). This is consistent with the sports-science reality that
  short-window injury prediction is hard; the inflated numbers common in this
  space tend to involve exactly the leaks this project caught and removed.

## Known limitations (stated up front)

- 74 athletes from one elite Dutch team: findings may not transfer to other
  populations, training cultures, or recreational runners
- Sex subgroup analysis is not supportable: the cohort is described as 27 women / 47
  men, but the released data maps no sex label to athlete IDs and we do not infer it
  (inferring sex from training volume would be methodologically wrong and unethical).
  Planned, found unsupportable, refused to proxy, documented - substituted tenure,
  volume-tier, and per-athlete-spread subgroups (see evaluation_design.md)
- Elite middle/long-distance running only; no strength/power athletes
- Observational data: the model finds risk associations, not causes; it cannot say
  "reduce load and injury risk drops"

## When NOT to trust the score

- (fill in from evaluation results: known blind spots, uncalibrated regions,
  populations outside the training distribution)
