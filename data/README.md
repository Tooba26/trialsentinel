# Data

The raw export (~420 MB) is not committed. To reproduce it:

1. Open the search at <https://clinicaltrials.gov/search>.
2. Filters:
   - **Study type:** Interventional
   - **Status:** Completed, Terminated, Withdrawn
   - **Phase:** Phase 2 and Phase 3 (this also returns Phase 1/2 and Phase 2/3 studies)
3. **Download → CSV**, all available fields, all results.
4. Unzip and save the file as `data/raw/ctg-studies.csv`.

The version used for the published model was downloaded on 28 October 2026 and contained 90,329 studies
(71,934 completed, 13,490 terminated, 4,905 withdrawn).

`data/processed/trials_features.parquet` is created by `make data`.

ClinicalTrials.gov data is provided by the U.S. National Library of Medicine and is in the public domain.
