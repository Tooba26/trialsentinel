from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

BASE = {
    "phase": "PHASE2", "funder_type": "INDUSTRY", "allocation": "RANDOMIZED",
    "intervention_model": "PARALLEL", "primary_purpose": "TREATMENT", "masking": "QUADRUPLE",
    "sex": "ALL", "age_groups": ["ADULT", "OLDER_ADULT"],
    "primary_condition": "Type 2 Diabetes", "intervention_types": ["DRUG"], "has_placebo": True,
    "n_interventions": 2, "n_sites": 40, "n_countries": 6, "has_us_site": True,
    "n_collaborators": 0, "n_conditions": 1, "n_primary_outcomes": 1, "n_secondary_outcomes": 5,
    "sponsor_prior_trials": 60, "sponsor_prior_termination_rate": 0.08,
    "registration_lag_days": 10, "start_year": 2019, "is_pilot": False,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_shape():
    r = client.post("/api/v1/predict", json=BASE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert body["risk_tier"] in {"Low", "Moderate", "High", "Critical"}
    assert 1 <= len(body["top_drivers"]) <= 3
    assert body["therapeutic_area"] == "Metabolic & endocrine"
    assert len(body["contributions"]) > 10


def test_risky_design_scores_higher():
    risky = {**BASE, "primary_condition": "Metastatic pancreatic cancer", "allocation": "NA",
             "intervention_model": "SINGLE_GROUP", "masking": "NONE", "has_placebo": False,
             "funder_type": "OTHER", "phase": "PHASE1|PHASE2", "sponsor_prior_trials": 3,
             "sponsor_prior_termination_rate": 0.6, "registration_lag_days": 400}
    p_safe = client.post("/api/v1/predict", json=BASE).json()["probability"]
    p_risky = client.post("/api/v1/predict", json=risky).json()["probability"]
    assert p_risky > p_safe


def test_sites_not_listed_is_accepted():
    r = client.post("/api/v1/predict", json={**BASE, "n_sites": None, "n_countries": None,
                                            "has_us_site": False})
    assert r.status_code == 200, r.text


def test_boundary_checks():
    assert client.post("/api/v1/predict", json={**BASE, "n_sites": -3}).status_code == 422
    assert client.post("/api/v1/predict", json={**BASE, "n_countries": 50}).status_code == 422
    assert client.post("/api/v1/predict", json={**BASE, "phase": "PHASE9"}).status_code == 422
    assert client.post("/api/v1/predict", json={**BASE, "unexpected": 1}).status_code == 422
    assert client.post("/api/v1/predict",
                       json={**BASE, "sponsor_prior_termination_rate": 1.5}).status_code == 422


def test_overview_and_sponsors():
    o = client.get("/api/v1/analytics/overview").json()
    assert o["headline"]["trials_analyzed"] > 50000
    assert "roc_auc" in o["model"]
    s = client.get("/api/v1/sponsors", params={"q": "pfizer"}).json()
    assert s and "termination_rate" in s[0]
