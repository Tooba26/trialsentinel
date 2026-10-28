"""Feature engineering shared by the training pipeline and the live API.

`engineer_from_raw` turns the ClinicalTrials.gov CSV export into one row of
registration-time features per trial. `add_engineered` derives the composite
indicators and is called by both training and inference, so the two paths can
never disagree.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config as C

_PILOT_RE = re.compile(r"\b(?:pilot|feasibility)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Small parsers
# ---------------------------------------------------------------------------
def _split(value: object, sep: str = "|") -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [p.strip() for p in value.split(sep) if p.strip()]


def parse_design(design: object) -> dict[str, str]:
    """'Allocation: RANDOMIZED | Intervention Model: PARALLEL | ...' -> dict."""
    out: dict[str, str] = {}
    for part in _split(design, "|"):
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip().lower()] = v.strip()
    return out


def masking_level(masking: str | None) -> float:
    if not masking:
        return np.nan
    head = masking.split("(")[0].strip().upper()
    return float(C.MASKING_LEVELS.get(head, np.nan))


def therapeutic_area(conditions: str) -> str:
    text = conditions.lower()
    for area, keywords in C.THERAPEUTIC_AREAS:
        if any(k in text for k in keywords):
            return area
    return "Other"


def normalize_condition(condition: str) -> str:
    c = re.sub(r"[^a-z0-9 ]+", " ", condition.lower())
    return re.sub(r"\s+", " ", c).strip() or "unknown"


def location_countries(locations: object) -> list[str]:
    return [loc.rsplit(",", 1)[-1].strip() for loc in _split(locations)]


def parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="mixed", errors="coerce")


# ---------------------------------------------------------------------------
# Composite indicators (shared with the API)
# ---------------------------------------------------------------------------
def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sites = df["n_sites"].astype(float)
    countries = df["n_countries"].astype(float)
    df["site_dispersion_index"] = np.where(sites > 0, countries / sites.replace(0, np.nan), np.nan)
    df["outcome_burden_score"] = df["n_primary_outcomes"].astype(float) + 0.5 * df["n_secondary_outcomes"].astype(float)
    df["eligibility_breadth"] = (
        df["age_child"].astype(float) + df["age_adult"].astype(float) + df["age_older"].astype(float)
        + (df["sex"] == "ALL").astype(float)
    )
    df["design_rigor_score"] = (
        (df["allocation"] == "RANDOMIZED").astype(float)
        + df["masking_level"].fillna(0).astype(float) / 4.0
        + df["has_placebo"].astype(float)
    )
    return df


# ---------------------------------------------------------------------------
# Sponsor history (time-respecting: only information available at start date)
# ---------------------------------------------------------------------------
def sponsor_history(raw: pd.DataFrame) -> pd.DataFrame:
    """For each trial: how many trials the sponsor started before it, and the
    termination rate among the sponsor's trials that had *finished* before it
    started. Uses every row of the export (all years, incl. withdrawn)."""
    d = pd.DataFrame({
        "sponsor": raw["Sponsor"].fillna("Unknown").str.strip(),
        "start": parse_date(raw["Start Date"]),
        "end": parse_date(raw["Completion Date"]).fillna(parse_date(raw["Primary Completion Date"])),
        "status": raw["Study Status"],
    }, index=raw.index)
    prior_trials = pd.Series(0.0, index=raw.index)
    prior_rate = pd.Series(np.nan, index=raw.index)

    for _, g in d.groupby("sponsor", sort=False):
        starts = g["start"].values
        valid_start = ~pd.isna(starts)
        sorted_starts = np.sort(starts[valid_start])
        finished = g[g["status"].isin(["COMPLETED", "TERMINATED"]) & g["end"].notna()].sort_values("end")
        ends = finished["end"].values
        cum_term = np.cumsum((finished["status"] == "TERMINATED").values)
        for idx, s in zip(g.index, starts):
            if pd.isna(s):
                prior_trials[idx] = np.nan
                continue
            prior_trials[idx] = np.searchsorted(sorted_starts, s, side="left")
            k = np.searchsorted(ends, s, side="left")  # finished strictly before start
            if k > 0:
                prior_rate[idx] = cum_term[k - 1] / k
    return pd.DataFrame({"sponsor_prior_trials": prior_trials, "sponsor_prior_termination_rate": prior_rate})


def sponsor_lookup_table(raw: pd.DataFrame, min_trials: int = 3) -> pd.DataFrame:
    """Current track record per sponsor, used by the API's sponsor search."""
    finished = raw[raw["Study Status"].isin(["COMPLETED", "TERMINATED"])]
    g = finished.groupby(finished["Sponsor"].str.strip())
    tbl = pd.DataFrame({
        "trials": g.size(),
        "termination_rate": g["Study Status"].apply(lambda s: (s == "TERMINATED").mean()),
    })
    tbl = tbl[tbl["trials"] >= min_trials].sort_values("trials", ascending=False)
    return tbl.reset_index().rename(columns={"Sponsor": "sponsor"})


# ---------------------------------------------------------------------------
# Raw CSV -> feature table
# ---------------------------------------------------------------------------
def engineer_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    design = raw["Study Design"].map(parse_design)
    conditions = raw["Conditions"].fillna("")
    interventions = raw["Interventions"].fillna("")
    int_types = interventions.map(lambda s: {p.split(":", 1)[0].strip().upper() for p in _split(s)})
    age = raw["Age"].fillna("")
    countries = raw["Locations"].map(location_countries)
    locations_missing = raw["Locations"].isna()
    start = parse_date(raw["Start Date"])
    first_posted = parse_date(raw["First Posted"])
    title_summary = raw["Study Title"].fillna("") + " " + raw["Brief Summary"].fillna("")

    f = pd.DataFrame(index=raw.index)
    f["nct_id"] = raw["NCT Number"]
    f["status"] = raw["Study Status"]
    f["phase"] = raw["Phases"]
    f["funder_type"] = raw["Funder Type"].fillna("UNKNOWN")
    f["allocation"] = design.map(lambda d: d.get("allocation"))
    f["intervention_model"] = design.map(lambda d: d.get("intervention model"))
    f["primary_purpose"] = design.map(lambda d: d.get("primary purpose"))
    f["masking_level"] = design.map(lambda d: masking_level(d.get("masking")))
    f["sex"] = raw["Sex"]
    f["therapeutic_area"] = conditions.map(therapeutic_area)
    f["primary_condition"] = conditions.map(lambda s: normalize_condition(_split(s)[0]) if _split(s) else "unknown")

    f["age_child"] = age.str.contains("CHILD").astype(int)
    f["age_adult"] = age.str.contains(r"(?<!OLDER_)ADULT").astype(int)
    f["age_older"] = age.str.contains("OLDER_ADULT").astype(int)

    for t in ["DRUG", "BIOLOGICAL", "DEVICE", "BEHAVIORAL", "PROCEDURE"]:
        f[f"has_{t.lower()}"] = int_types.map(lambda s, t=t: int(t in s))
    f["has_placebo"] = interventions.str.contains("placebo", case=False).astype(int)
    f["is_pilot"] = title_summary.str.contains(_PILOT_RE).astype(int)

    f["n_sites"] = countries.map(len).astype(float).where(~locations_missing)
    f["n_countries"] = countries.map(lambda c: len(set(c))).astype(float).where(~locations_missing)
    f["has_us_site"] = countries.map(lambda c: int("United States" in c))
    f["n_collaborators"] = raw["Collaborators"].map(lambda s: len(_split(s))).astype(float)
    f["n_conditions"] = conditions.map(lambda s: len(_split(s))).astype(float)
    f["n_interventions"] = interventions.map(lambda s: len(_split(s))).astype(float)
    f["n_primary_outcomes"] = raw["Primary Outcome Measures"].map(lambda s: len(_split(s))).astype(float)
    f["n_secondary_outcomes"] = raw["Secondary Outcome Measures"].map(lambda s: len(_split(s))).astype(float)

    f["start_year"] = start.dt.year.astype(float)
    f["registration_lag_days"] = (first_posted - start).dt.days.astype(float)
    f["start_date"] = start
    f["first_posted"] = first_posted

    f = f.join(sponsor_history(raw))
    f["sponsor"] = raw["Sponsor"].fillna("Unknown").str.strip()
    return add_engineered(f)
