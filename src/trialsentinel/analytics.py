"""Builds overview.json served by GET /api/v1/analytics/overview."""
from __future__ import annotations

import json

import pandas as pd

from . import config as C


def _rates(df: pd.DataFrame, col: str, min_n: int = 100, top: int | None = None) -> list[dict]:
    g = df.groupby(col)[C.TARGET].agg(["size", "mean"]).reset_index()
    g = g[g["size"] >= min_n].sort_values("mean", ascending=False)
    if top:
        g = g.head(top)
    return [{"group": str(r[col]), "trials": int(r["size"]), "termination_rate": round(float(r["mean"]), 4)}
            for _, r in g.iterrows()]


def run() -> None:
    df = pd.read_parquet(C.PROCESSED_DIR / "trials_features.parquet")
    meta = json.loads((C.ARTIFACTS_DIR / "model_metadata.json").read_text())
    quality = json.loads((C.ARTIFACTS_DIR / "data_quality_report.json").read_text())
    tm = meta["test_metrics"]
    df["sites_band"] = pd.cut(df["n_sites"], [0, 1, 5, 20, 100, 1e6],
                              labels=["1", "2-5", "6-20", "21-100", "100+"]).astype(str)
    df["sponsor_record_band"] = pd.cut(
        df["sponsor_prior_termination_rate"], [-0.01, 0.05, 0.10, 0.20, 1.0],
        labels=["0-5%", "5-10%", "10-20%", ">20%"]).astype(str).replace("nan", "No history")
    by_year = (df.groupby("start_year")[C.TARGET].agg(["size", "mean"]).reset_index())
    overview = {
        "headline": {
            "trials_analyzed": int(len(df)),
            "termination_rate": round(float(df[C.TARGET].mean()), 4),
            "terminated_trials": int(df[C.TARGET].sum()),
            "withdrawn_excluded": quality["withdrawn_excluded"],
            "start_year_window": meta["dataset"]["start_year_window"],
        },
        "model": {
            "selected": meta["selected_model"],
            "roc_auc": round(tm["roc_auc"], 4), "pr_auc": round(tm["pr_auc"], 4),
            "f1_macro": round(tm["f1_macro"], 4), "accuracy": round(tm["accuracy"], 4),
            "precision": round(tm["precision"], 4), "recall": round(tm["recall"], 4),
            "brier": round(tm["brier"], 4), "threshold": tm["threshold"],
            "top_decile_lift": round(tm["top_decile_lift"], 2),
            "top_decile_capture": round(tm["top_decile_capture"], 4),
            "temporal_roc_auc": round(meta["temporal_validation"]["roc_auc"], 4),
            "comparison": [{"model": k, "cv_pr_auc": round(v["cv_pr_auc"], 4),
                            "cv_roc_auc": round(v["cv_roc_auc"], 4),
                            "val_pr_auc": round(v["val_pr_auc"], 4),
                            "val_roc_auc": round(v["val_roc_auc"], 4)} for k, v in meta["models"].items()],
        },
        "risk_tiers": meta["risk_tiers"],
        "probability_histogram": meta.get("test_probability_histogram", []),
        "global_importance": [{**g, "label": C.FEATURE_LABELS.get(g["feature"], g["feature"])}
                              for g in meta["global_shap"][:12]],
        "segments": {
            "phase": _rates(df, "phase"),
            "therapeutic_area": _rates(df, "therapeutic_area"),
            "funder_type": _rates(df, "funder_type"),
            "sites": _rates(df, "sites_band", min_n=50),
            "sponsor_track_record": _rates(df, "sponsor_record_band", min_n=50),
            "intervention_model": _rates(df, "intervention_model"),
        },
        "by_start_year": [{"year": int(r["start_year"]), "trials": int(r["size"]),
                           "termination_rate": round(float(r["mean"]), 4)} for _, r in by_year.iterrows()],
        "data_quality": {"steps": quality["steps"], "excluded_for_leakage": quality["excluded_for_leakage"]},
    }
    (C.ARTIFACTS_DIR / "overview.json").write_text(json.dumps(overview, indent=2))
    print("wrote overview.json")


if __name__ == "__main__":
    run()
