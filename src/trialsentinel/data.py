"""Phase 1: ingestion, cleaning, sanity checks and the modeling table."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from .features import engineer_from_raw, sponsor_lookup_table


@dataclass
class QualityReport:
    steps: list[dict] = field(default_factory=list)

    def log(self, step: str, rows_before: int, rows_after: int, note: str = "") -> None:
        self.steps.append({"step": step, "rows_before": rows_before, "rows_after": rows_after,
                           "removed": rows_before - rows_after, "note": note})


def load_raw(path=C.RAW_CSV) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _cap_iqr(s: pd.Series, k: float = 3.0) -> tuple[pd.Series, int]:
    """IQR fences, widened to the 1st/99th percentiles so that zero-IQR
    count columns (e.g. most trials run in one country) are not flattened."""
    q1, q3, p01, p99 = s.quantile([0.25, 0.75, 0.01, 0.99])
    lo = min(q1 - k * (q3 - q1), p01)
    hi = max(q3 + k * (q3 - q1), p99)
    n = int(((s < lo) | (s > hi)).sum())
    return s.clip(lo, hi), n


def build_modeling_table(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    rep = QualityReport()
    n0 = len(raw)

    # --- Duplicates -------------------------------------------------------
    raw = raw.drop_duplicates()
    raw = raw.drop_duplicates(subset="NCT Number")
    rep.log("exact / NCT duplicates", n0, len(raw))
    key = raw["Study Title"].str.lower().str.strip() + "|" + raw["Sponsor"].fillna("") + "|" + raw["Start Date"].fillna("")
    n = len(raw)
    raw = raw[~key.duplicated()]
    rep.log("near-duplicates (same title, sponsor, start date)", n, len(raw),
            "multi-registrations of one protocol")

    # --- Features (computed on all rows so sponsor history sees everything) -
    feats = engineer_from_raw(raw)
    sponsors = sponsor_lookup_table(raw)

    # --- Structural anomalies ---------------------------------------------
    n = len(feats)
    bad_date = feats["start_date"].isna() | (feats["start_year"] < 1980) | (feats["start_year"] > 2027)
    feats = feats[~bad_date]
    rep.log("missing / impossible start dates", n, len(feats))
    n = len(feats)
    bad_posted = feats["first_posted"] < pd.Timestamp("1999-09-01")  # registry went live 2000
    feats = feats[~bad_posted]
    rep.log("first-posted date before the registry existed", n, len(feats))
    neg = (feats[["n_sites", "n_collaborators", "n_primary_outcomes"]] < 0).any(axis=1)
    rep.log("negative counts", len(feats), len(feats) - int(neg.sum()))
    feats = feats[~neg]

    # --- Scope: modeling window and target --------------------------------
    withdrawn = int((feats["status"] == "WITHDRAWN").sum())
    n = len(feats)
    feats = feats[feats["status"] != "WITHDRAWN"]
    rep.log("withdrawn trials (stopped before enrolling; separate outcome)", n, len(feats))
    n = len(feats)
    feats = feats[feats["start_year"].between(C.START_YEAR_MIN, C.START_YEAR_MAX)]
    rep.log(f"outside {C.START_YEAR_MIN}-{C.START_YEAR_MAX} start window (right-censoring)", n, len(feats))
    feats[C.TARGET] = (feats["status"] == "TERMINATED").astype(int)

    # --- Outlier capping (IQR, k=3) ---------------------------------------
    caps = {}
    for col in ["n_sites", "n_countries", "registration_lag_days", "n_secondary_outcomes",
                "n_primary_outcomes", "n_collaborators", "n_interventions", "n_conditions"]:
        z = (feats[col] - feats[col].mean()) / feats[col].std()
        feats[col], n_capped = _cap_iqr(feats[col])
        caps[col] = {"capped_iqr": n_capped, "abs_z_gt_3": int((z.abs() > 3).sum())}
    feats = feats.copy()
    from .features import add_engineered
    feats = add_engineered(feats)

    report = {
        "raw_rows": n0,
        "modeling_rows": len(feats),
        "withdrawn_excluded": withdrawn,
        "termination_rate": float(feats[C.TARGET].mean()),
        "steps": rep.steps,
        "outlier_capping": caps,
        "missing_share": {c: round(float(feats[c].isna().mean()), 4) for c in C.ALL_FEATURES},
        "excluded_for_leakage": [
            "Enrollment (export holds ACTUAL enrollment; terminated median 24 vs completed 90)",
            "Primary Completion Date / Completion Date (duration reveals early stop)",
            "Study Results, Results First Posted, Study Documents, Last Update Posted",
        ],
    }
    return feats.reset_index(drop=True), report, sponsors


def run() -> None:
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    C.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    feats, report, sponsors = build_modeling_table(raw)
    feats.to_parquet(C.PROCESSED_DIR / "trials_features.parquet", index=False)
    sponsors.to_json(C.ARTIFACTS_DIR / "sponsors.json", orient="records")
    (C.ARTIFACTS_DIR / "data_quality_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ["raw_rows", "modeling_rows", "termination_rate"]}, indent=2))
    for s in report["steps"]:
        print(f"  {s['step']}: -{s['removed']}")


if __name__ == "__main__":
    run()
