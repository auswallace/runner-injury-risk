# Assumptions & limitations

Living document - add a line the moment an assumption is made. Each entry: what we
assumed, why, and what breaks if it's wrong.

## Data assumptions

- [ ] Training logs are complete (are missing days "no training" or "not logged"?
      Check in EDA before assuming either)
- [ ] Injury labels mark true injury onset dates, not report dates
- [ ] Masked athlete IDs are stable across the full period

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
