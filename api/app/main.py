"""TrialSentinel inference API."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from trialsentinel import config as C

from .schemas import Prediction, Sponsor, TrialDesign
from .service import get_service


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_service()  # load the model once, before the first request
    yield


app = FastAPI(
    lifespan=lifespan,
    title="TrialSentinel API",
    version="1.0.0",
    description="Predicts whether a Phase 2/3 clinical trial will be terminated early, "
                "using only information available at registration, with SHAP explanations.",
)

origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                   allow_headers=["*"])


@app.get("/health")
def health() -> dict:
    s = get_service()
    return {"status": "ok", "model": s.model_name}


@app.post("/api/v1/predict", response_model=Prediction)
def predict(trial: TrialDesign) -> Prediction:
    """Termination probability, risk tier and the top 3 risk drivers for one trial design."""
    return get_service().predict(trial)


@app.get("/api/v1/analytics/overview")
def overview() -> dict:
    """Pre-computed cohort statistics, model performance and global feature importance."""
    return get_service().overview


@app.get("/api/v1/sponsors", response_model=list[Sponsor])
def sponsors(q: str = Query(min_length=2, max_length=100)) -> list[dict]:
    """Look up a sponsor's historical trial count and termination rate."""
    return get_service().search_sponsors(q)


@app.get("/api/v1/options")
def options() -> dict:
    """Allowed values for categorical inputs, for building forms."""
    return {"category_values": C.CATEGORY_VALUES, "masking": list(C.MASKING_LEVELS),
            "feature_labels": C.FEATURE_LABELS}
