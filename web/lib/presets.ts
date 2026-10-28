import type { TrialDesign } from "./types";

export const DEFAULT_DESIGN: TrialDesign = {
  phase: "PHASE2",
  funder_type: "INDUSTRY",
  allocation: "RANDOMIZED",
  intervention_model: "PARALLEL",
  primary_purpose: "TREATMENT",
  masking: "DOUBLE",
  sex: "ALL",
  age_groups: ["ADULT", "OLDER_ADULT"],
  primary_condition: "Type 2 diabetes",
  therapeutic_area: null,
  intervention_types: ["DRUG"],
  has_placebo: true,
  n_interventions: 2,
  n_sites: 25,
  n_countries: 3,
  has_us_site: true,
  n_collaborators: 0,
  n_conditions: 1,
  n_primary_outcomes: 1,
  n_secondary_outcomes: 4,
  sponsor_prior_trials: 40,
  sponsor_prior_termination_rate: 0.1,
  registration_lag_days: 14,
  start_year: 2021,
  is_pilot: false,
};

export const PRESETS: { name: string; description: string; design: TrialDesign }[] = [
  {
    name: "Large pharma Phase 3",
    description: "Randomized, quadruple-blind, 120 sites in 18 countries",
    design: {
      ...DEFAULT_DESIGN,
      phase: "PHASE3", masking: "QUADRUPLE", primary_condition: "Rheumatoid arthritis",
      n_sites: 120, n_countries: 18, sponsor_prior_trials: 600,
      sponsor_prior_termination_rate: 0.09, n_secondary_outcomes: 8,
      registration_lag_days: -20,
    },
  },
  {
    name: "Academic oncology pilot",
    description: "Single-arm, open-label, one hospital",
    design: {
      ...DEFAULT_DESIGN,
      phase: "PHASE1|PHASE2", funder_type: "OTHER", allocation: "NA",
      intervention_model: "SINGLE_GROUP", masking: "NONE", has_placebo: false,
      primary_condition: "Metastatic pancreatic cancer", n_interventions: 1,
      n_sites: 1, n_countries: 1, has_us_site: true, sponsor_prior_trials: 12,
      sponsor_prior_termination_rate: 0.3, registration_lag_days: 120, is_pilot: true,
      n_secondary_outcomes: 3,
    },
  },
  {
    name: "First-time biotech sponsor",
    description: "No track record, registered months after starting",
    design: {
      ...DEFAULT_DESIGN,
      primary_condition: "Alzheimer disease", masking: "DOUBLE",
      n_sites: 15, n_countries: 2, sponsor_prior_trials: 0,
      sponsor_prior_termination_rate: null, registration_lag_days: 240,
    },
  },
];
