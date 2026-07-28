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

- **Injury-day slice in the window.** The unsuffixed (most-recent-day) block is elevated
  on injury rows, i.e. it appears to include the injury day's own load. Keeping it
  matches the paper's same-day framing (good for benchmark comparison); dropping it is
  the honest choice for a *prospective* triage tool. Decide in 02/03 and label the task
  accordingly.
- **Sex subgroup source.** Neither table carries a sex/age column (only masked
  `Athlete ID`), yet the evaluation plan calls for a sex subgroup check. Either join an
  ID->sex mapping from the source paper's replication package, or drop the sex subgroup
  (keeping tenure/exposure, which *is* derivable) and state why. `evaluation_design.md`
  needs a footnote once decided.

## Modeling assumptions

- (add as made)

## Known limitations (stated up front)

- 74 athletes from one elite Dutch team: findings may not transfer to other
  populations, training cultures, or recreational runners
- Only 27 women in the sample - subgroup estimates for women are noisy
- Elite middle/long-distance running only; no strength/power athletes
- Observational data: the model finds risk associations, not causes; it cannot say
  "reduce load and injury risk drops"

## When NOT to trust the score

- (fill in from evaluation results: known blind spots, uncalibrated regions,
  populations outside the training distribution)
