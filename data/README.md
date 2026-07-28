# Data provenance

**Source:** Replication Data for "Injury Prediction in Competitive Runners With
Machine Learning" - University of Groningen, DataverseNL.
DOI: [10.34894/uwu9pv](https://doi.org/10.34894/uwu9pv). Open access.

**Contents:** ~7 years (2012-2019) of day-by-day training logs for 74 competitive
middle/long-distance runners (27 women, 47 men) from a Dutch elite team, with
injury event labels. Ships in two framings: a day-approach table and a
week-approach table (rolling aggregates).

**Citation:** Lovdal, S., den Hartigh, R., & Azzopardi, G. (2021). Injury
Prediction in Competitive Runners With Machine Learning. *IJSPP*, 16(10),
1522-1531. https://doi.org/10.1123/ijspp.2020-0518

**Rules for this repo:**
- Raw files stay in `data/raw/` and are **never committed** (see .gitignore);
  anyone can re-fetch with `python data/fetch_data.py`
- The data is anonymized by the publishers (masked athlete IDs); keep it that way -
  no attempts to re-identify, no joining with other athlete data
- Any derived/processed tables that get committed must be aggregates that cannot
  reconstruct an individual athlete's log
