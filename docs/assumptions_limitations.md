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

- **Injury-day slice in the window.** Block ordering was proven in 01_eda (2.5M
  zero-mismatch slide checks): **`.6` is the most recent day - the injury day on
  positive rows - and the unsuffixed block is the oldest (t-6).** An earlier draft had
  this backwards from a univariate load comparison; the slide check settled it.
  Keeping the `.6` slice matches the paper's same-day framing (good for benchmark
  comparison); dropping it is the honest choice for a *prospective* triage tool.
  Decide in 02/03 and label the task accordingly.
- ~~**Sex subgroup source.**~~ *Settled:* no sex label ships with the data and we will
  not infer it; sex subgroup replaced by tenure / volume-tier / per-athlete-spread
  checks and recorded as a limitation. See Known limitations above and
  `evaluation_design.md`. (Open confirmation for Austin: verify the Dataverse original
  ships no athlete-attributes/codebook file the GitHub mirror dropped - the sandbox
  can't reach Dataverse.)

## Modeling assumptions

- (add as made)

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
