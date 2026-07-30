# Runner Injury Risk Model

Predicting injury risk in competitive runners from day-by-day training load, on the
University of Groningen open dataset (74 elite Dutch runners, 7 years of training logs,
injury event labels).

**Why this project:** my background is security data analytics, and injury-risk
surveillance turns out to be the same shape of problem I work on every day: rare
events buried in messy longitudinal data, and an alert stream that people stop
trusting the moment it cries wolf. This repo builds the full loop on public data:
features from training load, a calibrated classifier, honest evaluation against a
published benchmark, and a decision-support view of the output.

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
dashboard/       decision-support view builder (generated HTML gitignored)
docs/            evaluation design, assumptions & limitations, session log
```

## Method

- Features: 12 seven-day training-load aggregates (distance, intensity mix, rest,
  strength work, perceived exertion and recovery), each kept in raw form and also
  normalized against that athlete's own earlier healthy days. I planned chronic
  load and ACWR features too. They died on a dataset artifact, and finding out why
  taught me more than the features would have - that story is under Status
- Models: regularized logistic regression first, then XGBoost (the source paper
  used bagged XGBoost, so its reported AUC is the benchmark). Logistic won every
  comparison; the fancier model never earned its keep here
- Evaluation: a time-aware split (train on early years, test on later ones) and an
  athlete-grouped split (test athletes fully unseen), reported separately.
  Precision-recall leads because ROC flatters rare events. Calibration gets checked
  with Brier scores, and there is an alert-budget table because "how many flags per
  week can my staff handle" is the question a real program would actually ask. I
  wrote the evaluation design down in `docs/evaluation_design.md` before modeling
  so I could not quietly bend it later
- Output: a weekly flagged-athletes view with the top signals behind each flag and
  a "when not to trust this" panel

## Status

- 01_eda done: 1.36% positive rate (583/42,766 rows), Date confirmed as a shared
  calendar index, missingness resolved (zero-km days are logged rest; true gaps live
  between rows), window ordering proven structurally (2.5M zero-mismatch checks)
- 02_features + 03_model done, and the two corrections along the way are the most
  interesting part of the project:
  1. **A first feature set with 28-day chronic load / ACWR hit validation AUC 0.987
     and was killed as a leak.** Those features were NaN for 100% of injury rows
     versus 4-19% of healthy rows, and a single missing-indicator flag carried 93%
     of XGBoost's gain. Cause: the data's publishers required healthy events to be
     injury-free for 3 weeks either side, so any window reaching past a row's own 7
     days is empty for injury rows only and its missingness encodes the label.
     Confirmed afterwards in the paper's own Methods. Chronic load and ACWR are
     therefore unbuildable on this dataset - documented rather than shipped.
  2. **Reading the source paper corrected a window assumption of mine.** All 7 slots
     are pre-event days (the target is whether the *next* session brings injury), so
     the newest slot is the day before the injury, not the injury day. An
     over-cautious 6-day window was costing real signal
- Leak-free results, closing on the published benchmark: per-athlete normalization
  (adopted from the paper, recomputed to use only each athlete's earlier healthy
  rows so no row sees its own future) roughly doubles average precision. Best model
  is regularized logistic regression - it beat XGBoost on every comparison
- Athlete-grouped CV: **AUC 0.690, AP 0.064 vs 0.014 chance (4.5x)**, against the
  paper's 0.724 on their closest-analogue split (their test set is the 10 newest
  athletes). Time-aware: AUC 0.634, AP 0.032 vs 0.016 chance
- The paper reports ROC only. At this dataset's 1.4% prevalence its published
  operating point (58.4% sensitivity, 74.1% specificity) implies roughly 3%
  precision - about 32 false alarms per true flag. Making that visible, via
  precision-recall, calibration and an alert-budget view, is what 04 adds
- 04_evaluation done. I kept the test sets locked until every decision
  (calibration method, thresholds, subgroup cutoffs) was settled on validation
  data, then evaluated once and stopped:
  - Test AP 0.030 on both splits (1.7x chance on the time-aware split, 2.6x on
    the athlete-grouped one); AUC 0.629 / 0.619 against the paper's 0.724
  - Grouped CV had said 0.690, so the drop to 0.619 sent me looking. Per-athlete
    AUC across the 18 held-out runners goes from 0.27 to 0.83, and 2 of 15 sit
    below a coin flip. The average was hiding athletes the model gets actively
    backwards. This is the exact failure the per-athlete check in my evaluation
    design existed to catch, and it caught it
  - Calibration was a lesson. Platt scaling fixes the absurd raw probabilities
    that class-weighted training produces (Brier 0.245 down to 0.017), but the
    calibrated score still cannot beat just predicting the 1.4% base rate every
    time. The model earns its keep by ranking, not by putting a believable
    percentage on anyone. That finding is why the dashboard shows risk bands
    instead of probabilities
  - The alert-budget table: at 1 to 5 flags per week, precision is 2.9-4.3% and
    recall is 2-18%. That sounds rough until you work out that the paper's own
    published operating point implies about 3% precision too. They never printed
    that number, and I think it is the most important one in this problem
  - Flagging two days ahead instead of one costs almost nothing (grouped AP 0.032
    vs 0.030). Good news, since a one-day warning leaves little time to act
- Decision-support dashboard done (see below)
- Findings and every assumption logged in `docs/assumptions_limitations.md`,
  including the filled-in "when NOT to trust the score" section

## Decision-support view

A model number is not a decision, and this was the part of the project I most
wanted to get right. The dashboard is one HTML file, no server and no
dependencies, that replays the held-out final period week by week the way staff
would have lived it: pick how many flags per week your people can act on, see who
gets flagged and which signals drove each flag, and read the model's blind spots
on the same page as its output.

![Dashboard overview](docs/img/dashboard_overview.png)

![Weekly flagged-athletes view](docs/img/dashboard_week_view.png)

Two choices came straight out of the evaluation. Risk shows as bands, never as a
percentage, because I now know the calibrated probabilities are no sharper than
the base rate. And nearly every top signal behind a flag turns out to be a "vs
own history" feature, which backs up the modeling result: trouble does not look
like big mileage, it looks like a runner drifting away from their own normal.

The generated file embeds athlete-day rows, so it stays out of git (rule in
`data/README.md`). Rebuild it locally with:

```
python3 dashboard/build_dashboard.py
```

The builder refits the exact model that was evaluated and refuses to write the
file if the refit does not reproduce the saved test metrics. I did not want a
dashboard that could quietly drift away from the numbers I published.
