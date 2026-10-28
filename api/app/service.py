"""Model loading and inference: preprocessing, calibrated probability, local SHAP."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from trialsentinel import config as C
from trialsentinel.features import add_engineered, normalize_condition, therapeutic_area

from .schemas import Driver, Prediction, TrialDesign

ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", C.ARTIFACTS_DIR))


class ModelService:
    def __init__(self, artifacts: Path = ARTIFACTS) -> None:
        b = joblib.load(artifacts / "model_bundle.joblib")
        self.pre = b["preprocessor"]
        self.model = b["model"]
        self.calibrator = b["calibrator"]
        self.model_name = b["model_name"]
        self.features = b["features"]
        self.source_map = b["source_map"]
        self.threshold = b["threshold"]
        self.tiers = b["risk_tiers"]
        meta = json.loads((artifacts / "model_metadata.json").read_text())
        self.base_rate = meta["dataset"]["termination_rate"]
        self.metadata = meta
        self.overview = json.loads((artifacts / "overview.json").read_text())
        self.sponsors = json.loads((artifacts / "sponsors.json").read_text())
        if self.model_name == "logreg":
            bg = np.load(artifacts / "logreg_background.npy")
            self.explainer = shap.LinearExplainer(self.model, bg)
        else:
            self.explainer = shap.TreeExplainer(self.model)
        ev = self.explainer.expected_value
        self.expected_value = float(np.ravel(ev)[-1])

    # ------------------------------------------------------------------
    def to_frame(self, t: TrialDesign) -> tuple[pd.DataFrame, str]:
        area = t.therapeutic_area or therapeutic_area(t.primary_condition)
        ages = set(t.age_groups)
        types = set(t.intervention_types)
        row = {
            "phase": t.phase, "funder_type": t.funder_type, "allocation": t.allocation,
            "intervention_model": t.intervention_model, "primary_purpose": t.primary_purpose,
            "sex": t.sex, "therapeutic_area": area,
            "primary_condition": normalize_condition(t.primary_condition),
            "age_child": int("CHILD" in ages), "age_adult": int("ADULT" in ages),
            "age_older": int("OLDER_ADULT" in ages),
            "has_drug": int("DRUG" in types), "has_biological": int("BIOLOGICAL" in types),
            "has_device": int("DEVICE" in types), "has_behavioral": int("BEHAVIORAL" in types),
            "has_procedure": int("PROCEDURE" in types), "has_placebo": int(t.has_placebo),
            "has_us_site": int(t.has_us_site), "is_pilot": int(t.is_pilot),
            "n_collaborators": t.n_collaborators, "n_conditions": t.n_conditions,
            "n_interventions": t.n_interventions, "n_primary_outcomes": t.n_primary_outcomes,
            "n_secondary_outcomes": t.n_secondary_outcomes,
            "sponsor_prior_trials": t.sponsor_prior_trials,
            "n_sites": np.nan if t.n_sites is None else t.n_sites,
            "n_countries": np.nan if t.n_countries is None else t.n_countries,
            "sponsor_prior_termination_rate": (np.nan if t.sponsor_prior_termination_rate is None
                                               else t.sponsor_prior_termination_rate),
            "registration_lag_days": t.registration_lag_days,
            "masking_level": C.MASKING_LEVELS[t.masking], "start_year": t.start_year,
        }
        df = add_engineered(pd.DataFrame([row]))
        return df[self.features], area

    def risk_tier(self, p: float) -> str:
        return next(name for name, lo, hi in self.tiers if lo <= p < hi)

    @staticmethod
    def _display(feature: str, value) -> str:
        if isinstance(value, float) and np.isnan(value):
            return "not listed"
        if feature == "sponsor_prior_termination_rate":
            return f"{value:.0%}"
        if feature.startswith(("has_", "age_", "is_")):
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:g}" if value == int(value) else f"{value:.2f}"
        return str(value)

    def predict(self, t: TrialDesign) -> Prediction:
        X, area = self.to_frame(t)
        Xt = self.pre.transform(X)
        p = float(self.calibrator.predict_proba(Xt)[0, 1])
        sv = np.asarray(self.explainer.shap_values(Xt))
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        contrib = pd.Series(sv[0], index=self.source_map).groupby(level=0).sum()
        contrib = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
        drivers = [
            Driver(feature=f, label=C.FEATURE_LABELS.get(f, f), value=self._display(f, X.iloc[0][f]),
                   contribution=round(float(v), 4),
                   direction="raises risk" if v > 0 else "lowers risk")
            for f, v in contrib.items()
        ]
        warnings = []
        if t.start_year > C.START_YEAR_MAX:
            warnings.append(f"Model trained on trials started {C.START_YEAR_MIN}-{C.START_YEAR_MAX}; "
                            "later start years are treated like the most recent ones.")
        if X.iloc[0]["primary_condition"] not in self._known_conditions():
            warnings.append("Condition not seen in training; its effect falls back to the average.")
        return Prediction(
            probability=round(p, 4), risk_tier=self.risk_tier(p), flagged=p >= self.threshold,
            threshold=self.threshold, base_rate=round(self.base_rate, 4),
            relative_risk=round(p / self.base_rate, 2), therapeutic_area=area,
            top_drivers=self._top_drivers(drivers),
            contributions=drivers, expected_value=round(self.expected_value, 4),
            model=self.model_name, warnings=warnings,
        )

    # Start year is an era effect, not a design lever: it stays in `contributions`
    # but is never presented as one of the headline risk drivers.
    NON_ACTIONABLE = frozenset({"start_year"})

    def _top_drivers(self, drivers: list[Driver]) -> list[Driver]:
        actionable = [d for d in drivers if d.feature not in self.NON_ACTIONABLE]
        raising = [d for d in actionable if d.contribution > 0]
        return raising[:3] or actionable[:3]

    @lru_cache(maxsize=1)
    def _known_conditions(self) -> frozenset:
        te = self.pre.named_transformers_["te"].named_steps["target"]
        return frozenset(te.categories_[0])

    def search_sponsors(self, q: str, limit: int = 8) -> list[dict]:
        q = q.lower().strip()
        hits = [s for s in self.sponsors if q in s["sponsor"].lower()]
        return hits[:limit]


@lru_cache(maxsize=1)
def get_service() -> ModelService:
    return ModelService()
