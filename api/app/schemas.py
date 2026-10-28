"""Strictly typed request/response models with boundary checks."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from trialsentinel import config as C

Phase = Literal["PHASE2", "PHASE3", "PHASE1|PHASE2", "PHASE2|PHASE3"]
FunderType = Literal["INDUSTRY", "OTHER", "NIH", "NETWORK", "OTHER_GOV", "FED", "INDIV", "UNKNOWN"]
Allocation = Literal["RANDOMIZED", "NON_RANDOMIZED", "NA"]
InterventionModel = Literal["PARALLEL", "SINGLE_GROUP", "CROSSOVER", "FACTORIAL", "SEQUENTIAL"]
Purpose = Literal["TREATMENT", "PREVENTION", "SUPPORTIVE_CARE", "DIAGNOSTIC", "BASIC_SCIENCE",
                  "HEALTH_SERVICES_RESEARCH", "SCREENING", "DEVICE_FEASIBILITY", "OTHER"]
Masking = Literal["NONE", "SINGLE", "DOUBLE", "TRIPLE", "QUADRUPLE"]
Sex = Literal["ALL", "FEMALE", "MALE"]
AgeGroup = Literal["CHILD", "ADULT", "OLDER_ADULT"]
InterventionType = Literal["DRUG", "BIOLOGICAL", "DEVICE", "BEHAVIORAL", "PROCEDURE",
                           "DIETARY_SUPPLEMENT", "RADIATION", "GENETIC", "COMBINATION_PRODUCT", "OTHER"]
TherapeuticArea = Literal[tuple(C.THERAPEUTIC_AREA_VALUES)]  # type: ignore[valid-type]


class TrialDesign(BaseModel):
    """A trial as it would be described at registration time."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "phase": "PHASE2", "funder_type": "INDUSTRY", "allocation": "RANDOMIZED",
        "intervention_model": "PARALLEL", "primary_purpose": "TREATMENT", "masking": "QUADRUPLE",
        "sex": "ALL", "age_groups": ["ADULT", "OLDER_ADULT"],
        "primary_condition": "Non-small cell lung cancer", "therapeutic_area": None,
        "intervention_types": ["DRUG"], "has_placebo": True, "n_interventions": 2,
        "n_sites": 40, "n_countries": 6, "has_us_site": True, "n_collaborators": 0,
        "n_conditions": 1, "n_primary_outcomes": 1, "n_secondary_outcomes": 5,
        "sponsor_prior_trials": 60, "sponsor_prior_termination_rate": 0.15,
        "registration_lag_days": 10, "start_year": 2021, "is_pilot": False}})

    phase: Phase
    funder_type: FunderType
    allocation: Allocation
    intervention_model: InterventionModel
    primary_purpose: Purpose
    masking: Masking
    sex: Sex
    age_groups: list[AgeGroup] = Field(min_length=1, max_length=3)
    primary_condition: str = Field(min_length=2, max_length=200)
    therapeutic_area: Optional[TherapeuticArea] = Field(
        default=None, description="Leave empty to infer it from the condition text.")
    intervention_types: list[InterventionType] = Field(min_length=1, max_length=10)
    has_placebo: bool
    n_interventions: int = Field(ge=1, le=30)
    n_sites: Optional[int] = Field(default=None, ge=1, le=2000,
                                   description="Null when sites are not yet listed.")
    n_countries: Optional[int] = Field(default=None, ge=1, le=100)
    has_us_site: bool
    n_collaborators: int = Field(ge=0, le=50)
    n_conditions: int = Field(ge=1, le=50)
    n_primary_outcomes: int = Field(ge=1, le=30)
    n_secondary_outcomes: int = Field(ge=0, le=100)
    sponsor_prior_trials: int = Field(ge=0, le=5000)
    sponsor_prior_termination_rate: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Null when the sponsor has no finished trials.")
    registration_lag_days: int = Field(ge=-3650, le=3650,
                                       description="Days from start date to registration; negative = registered before start.")
    start_year: int = Field(ge=2000, le=2035)
    is_pilot: bool = False

    @field_validator("primary_condition")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("primary_condition is too short")
        return v

    @model_validator(mode="after")
    def _consistency(self) -> "TrialDesign":
        if (self.n_sites is None) != (self.n_countries is None):
            raise ValueError("n_sites and n_countries must both be set or both be null")
        if self.n_sites is not None and self.n_countries > self.n_sites:
            raise ValueError("n_countries cannot exceed n_sites")
        if self.has_us_site and self.n_sites is None:
            raise ValueError("has_us_site requires sites to be listed")
        if self.has_placebo and self.allocation != "RANDOMIZED":
            raise ValueError("a placebo arm requires randomized allocation")
        if self.sponsor_prior_trials == 0 and self.sponsor_prior_termination_rate is not None:
            raise ValueError("a sponsor with no prior trials cannot have a track record")
        return self


class Driver(BaseModel):
    feature: str
    label: str
    value: str
    contribution: float = Field(description="SHAP contribution in log-odds; >0 raises risk")
    direction: Literal["raises risk", "lowers risk"]


class Prediction(BaseModel):
    probability: float
    risk_tier: Literal["Low", "Moderate", "High", "Critical"]
    flagged: bool
    threshold: float
    base_rate: float
    relative_risk: float
    therapeutic_area: str
    top_drivers: list[Driver]
    contributions: list[Driver]
    expected_value: float
    model: str
    warnings: list[str] = []


class Sponsor(BaseModel):
    sponsor: str
    trials: int
    termination_rate: float
