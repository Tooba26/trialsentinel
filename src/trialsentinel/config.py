"""Central configuration for the TrialSentinel pipeline."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "raw" / "ctg-studies.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
FIGURES_DIR = ROOT / "reports" / "figures"

RANDOM_STATE = 42

# Modeling window. Trials that started after 2021 are mostly still running; the
# ones already finished in the export are a biased (fast-finishing) sample.
START_YEAR_MIN = 2000
START_YEAR_MAX = 2021

TARGET = "terminated"

# ---------------------------------------------------------------------------
# Feature groups. Every feature here is knowable when a trial is registered.
# Deliberately EXCLUDED (leakage): Enrollment (actual, not planned), Completion
# dates / duration, Study Results, Results First Posted, Last Update Posted,
# Study Documents, and any "why stopped" text.
# ---------------------------------------------------------------------------
LOW_CARD_CATEGORICAL = [
    "phase",
    "funder_type",
    "allocation",
    "intervention_model",
    "primary_purpose",
    "sex",
    "therapeutic_area",
]
HIGH_CARD_CATEGORICAL = ["primary_condition"]  # target-encoded
BINARY = [
    "age_child",
    "age_adult",
    "age_older",
    "has_drug",
    "has_biological",
    "has_device",
    "has_behavioral",
    "has_procedure",
    "has_placebo",
    "has_us_site",
    "is_pilot",
]
# Skewed counts: median imputation + log1p.
SKEWED_NUMERIC = [
    "n_collaborators",
    "n_conditions",
    "n_interventions",
    "n_primary_outcomes",
    "n_secondary_outcomes",
    "sponsor_prior_trials",
]
# Missing-not-at-random numerics: IterativeImputer (MICE) + missing indicators.
MNAR_NUMERIC = [
    "n_sites",
    "n_countries",
    "sponsor_prior_termination_rate",
    "registration_lag_days",
]
ORDINAL_NUMERIC = ["masking_level", "start_year"]
# Engineered composite indicators.
ENGINEERED = [
    "site_dispersion_index",
    "outcome_burden_score",
    "eligibility_breadth",
    "design_rigor_score",
]

ALL_FEATURES = (
    LOW_CARD_CATEGORICAL
    + HIGH_CARD_CATEGORICAL
    + BINARY
    + SKEWED_NUMERIC
    + MNAR_NUMERIC
    + ORDINAL_NUMERIC
    + ENGINEERED
)

CATEGORY_VALUES = {
    "phase": ["PHASE2", "PHASE3", "PHASE1|PHASE2", "PHASE2|PHASE3"],
    "funder_type": ["INDUSTRY", "OTHER", "NIH", "NETWORK", "OTHER_GOV", "FED", "INDIV", "UNKNOWN"],
    "allocation": ["RANDOMIZED", "NON_RANDOMIZED", "NA", "Unknown"],
    "intervention_model": ["PARALLEL", "SINGLE_GROUP", "CROSSOVER", "FACTORIAL", "SEQUENTIAL", "Unknown"],
    "primary_purpose": [
        "TREATMENT", "PREVENTION", "SUPPORTIVE_CARE", "DIAGNOSTIC", "BASIC_SCIENCE",
        "HEALTH_SERVICES_RESEARCH", "SCREENING", "DEVICE_FEASIBILITY", "OTHER", "Unknown",
    ],
    "sex": ["ALL", "FEMALE", "MALE", "Unknown"],
}

MASKING_LEVELS = {"NONE": 0, "SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "QUADRUPLE": 4}

# Keyword rules mapping free-text conditions to a therapeutic area. Order matters:
# the first matching area wins.
THERAPEUTIC_AREAS: list[tuple[str, tuple[str, ...]]] = [
    ("Oncology", ("cancer", "tumor", "tumour", "carcinoma", "lymphoma", "leukemia", "leukaemia",
                  "melanoma", "myeloma", "neoplasm", "sarcoma", "glioma", "glioblastoma",
                  "malignan", "metasta", "oncolog", "blastoma", "mesothelioma")),
    ("Infectious disease", ("hiv", "hepatitis", "infection", "covid", "sars", "influenza",
                            "tuberculosis", "malaria", "vaccine", "sepsis", "pneumonia", "bacterial",
                            "viral", "virus", "fungal", "hpv")),
    ("Cardiovascular", ("heart", "cardiac", "cardio", "hypertension", "coronary", "atrial",
                        "myocardial", "stroke", "thrombo", "vascular", "arrhythm", "angina")),
    ("Metabolic & endocrine", ("diabetes", "obesity", "diabetic", "insulin", "lipid", "cholesterol",
                               "thyroid", "metabolic", "hypoglyc", "growth hormone")),
    ("Neurology", ("alzheimer", "parkinson", "epilep", "seizure", "multiple sclerosis", "migraine",
                   "neuropath", "dementia", "amyotrophic", "huntington", "neuro")),
    ("Psychiatry", ("depress", "schizophren", "bipolar", "anxiety", "adhd", "autism", "psych",
                    "alcohol", "opioid use", "substance", "smoking", "insomnia", "ptsd")),
    ("Respiratory", ("asthma", "copd", "pulmonary", "cystic fibrosis", "respiratory", "lung disease",
                     "bronch")),
    ("Immunology & rheumatology", ("arthritis", "lupus", "psoria", "crohn", "colitis", "inflammatory bowel",
                                   "spondyl", "atopic", "eczema", "allerg", "urticaria", "sjogren")),
    ("Pain & anesthesia", ("pain", "anesthe", "anaesthe", "analges", "postoperative", "nausea")),
    ("Renal & urology", ("kidney", "renal", "urinary", "bladder", "prostat", "dialysis")),
    ("Hematology", ("anemia", "anaemia", "hemophilia", "haemophilia", "sickle", "thalass",
                    "neutropenia", "platelet")),
    ("Ophthalmology", ("macular", "glaucoma", "retin", "eye", "ocular", "uveitis", "cornea")),
    ("Gastroenterology & hepatology", ("liver", "cirrhosis", "gastro", "bowel", "constipation",
                                       "pancrea", "reflux", "nash", "steatohep")),
    ("Women's health", ("pregnan", "contracep", "endometriosis", "menopaus", "infertil", "preterm",
                        "ovarian syndrome", "labor")),
    ("Dermatology", ("acne", "skin", "dermat", "wound", "alopecia", "rosacea")),
]
THERAPEUTIC_AREA_VALUES = [a for a, _ in THERAPEUTIC_AREAS] + ["Other"]
CATEGORY_VALUES["therapeutic_area"] = THERAPEUTIC_AREA_VALUES

# Human-readable names used by the API and UI for SHAP drivers.
FEATURE_LABELS = {
    "phase": "Trial phase",
    "funder_type": "Funder type",
    "allocation": "Allocation",
    "intervention_model": "Intervention model",
    "primary_purpose": "Primary purpose",
    "sex": "Sex eligibility",
    "therapeutic_area": "Therapeutic area",
    "primary_condition": "Primary condition",
    "age_child": "Enrolls children",
    "age_adult": "Enrolls adults",
    "age_older": "Enrolls older adults",
    "has_drug": "Drug intervention",
    "has_biological": "Biological intervention",
    "has_device": "Device intervention",
    "has_behavioral": "Behavioral intervention",
    "has_procedure": "Procedure intervention",
    "has_placebo": "Placebo arm",
    "has_us_site": "Has a US site",
    "is_pilot": "Pilot / feasibility study",
    "n_collaborators": "Number of collaborators",
    "n_conditions": "Number of conditions",
    "n_interventions": "Number of interventions",
    "n_primary_outcomes": "Primary outcomes",
    "n_secondary_outcomes": "Secondary outcomes",
    "sponsor_prior_trials": "Sponsor experience (prior trials)",
    "n_sites": "Number of sites",
    "n_countries": "Number of countries",
    "sponsor_prior_termination_rate": "Sponsor track record (past termination rate)",
    "registration_lag_days": "Registration lag (days after start)",
    "masking_level": "Blinding level",
    "start_year": "Start year (era effect)",
    "site_dispersion_index": "Site dispersion index",
    "outcome_burden_score": "Outcome burden score",
    "eligibility_breadth": "Eligibility breadth",
    "design_rigor_score": "Design rigor score",
}
