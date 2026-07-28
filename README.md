# Runner Injury Risk Model

Predicting injury risk in competitive runners from day-by-day training load, on the
University of Groningen open dataset (74 elite Dutch runners, 7 years of training logs,
injury event labels).

**Why this project:** injury-risk surveillance is a rare-event prediction problem with
real operational stakes: messy longitudinal data, heavy class imbalance, and a score
that has to be trustworthy enough to drive preventive decisions about people. This
repo builds that loop end to end: features from training load, a calibrated classifier,
honest evaluation against a published benchmark, and a decision-support view of the
output.

*Independent project on public data. Not affiliated with any military program,
government agency, or employer.*

## Data

Lovdal, S., den Hartigh, R., & Azzopardi, G. (2021). Injury Prediction in Competitive
Runners With Machine Learning. *International Journal of Sports Physiology and
Performance*, 16(10), 1522-1531. Replication data: DOI
[10.34894/uwu9pv](https://doi.org/10.34894/uwu9pv) (DataverseNL, open access).

Run `python data/fetch_data.py` to download the data into `data/raw/` (not committed
to git). See `data/README.md` for provenance and citation details.

## Structure

```
data/            fetch script + provenance notes (raw data gitignored)
notebooks/       01 EDA -> 02 features -> 03 model -> 04 evaluation
docs/            evaluation design, assumptions & limitations
```

## Method (planned)

- Features: acute (7-day) vs chronic (28-day) training load and their ratio, session
  intensity distribution, recovery patterns
- Models: regularized logistic regression baseline, then tree ensembles (the source
  paper used bagged XGBoost - its reported AUC is the benchmark)
- Evaluation: time-aware splits (train early years, test later) and athlete-grouped
  splits to prevent leakage; precision-recall alongside ROC (rare events make ROC
  flattering); calibration curves; subgroup checks
- Output: weekly flagged-athletes view with top contributing features and an explicit
  "when not to trust this score" panel

## Status

Scaffold stage. EDA in progress.
