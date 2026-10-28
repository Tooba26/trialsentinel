export type Phase = "PHASE2" | "PHASE3" | "PHASE1|PHASE2" | "PHASE2|PHASE3";
export type Masking = "NONE" | "SINGLE" | "DOUBLE" | "TRIPLE" | "QUADRUPLE";
export type AgeGroup = "CHILD" | "ADULT" | "OLDER_ADULT";
export type InterventionType =
  | "DRUG" | "BIOLOGICAL" | "DEVICE" | "BEHAVIORAL" | "PROCEDURE"
  | "DIETARY_SUPPLEMENT" | "RADIATION" | "GENETIC" | "OTHER";

export interface TrialDesign {
  phase: Phase;
  funder_type: string;
  allocation: "RANDOMIZED" | "NON_RANDOMIZED" | "NA";
  intervention_model: string;
  primary_purpose: string;
  masking: Masking;
  sex: "ALL" | "FEMALE" | "MALE";
  age_groups: AgeGroup[];
  primary_condition: string;
  therapeutic_area: string | null;
  intervention_types: InterventionType[];
  has_placebo: boolean;
  n_interventions: number;
  n_sites: number | null;
  n_countries: number | null;
  has_us_site: boolean;
  n_collaborators: number;
  n_conditions: number;
  n_primary_outcomes: number;
  n_secondary_outcomes: number;
  sponsor_prior_trials: number;
  sponsor_prior_termination_rate: number | null;
  registration_lag_days: number;
  start_year: number;
  is_pilot: boolean;
}

export interface Driver {
  feature: string;
  label: string;
  value: string;
  contribution: number;
  direction: "raises risk" | "lowers risk";
}

export interface Prediction {
  probability: number;
  risk_tier: "Low" | "Moderate" | "High" | "Critical";
  flagged: boolean;
  threshold: number;
  base_rate: number;
  relative_risk: number;
  therapeutic_area: string;
  top_drivers: Driver[];
  contributions: Driver[];
  expected_value: number;
  model: string;
  warnings: string[];
}

export interface Rate { group: string; trials: number; termination_rate: number }

export interface Overview {
  headline: {
    trials_analyzed: number;
    termination_rate: number;
    terminated_trials: number;
    withdrawn_excluded: number;
    start_year_window: [number, number];
  };
  model: {
    selected: string;
    roc_auc: number; pr_auc: number; f1_macro: number; accuracy: number;
    precision: number; recall: number; brier: number; threshold: number;
    top_decile_lift: number; top_decile_capture: number; temporal_roc_auc: number;
    comparison: { model: string; cv_pr_auc: number; cv_roc_auc: number; val_pr_auc: number; val_roc_auc: number }[];
  };
  risk_tiers: { tier: string; range: [number, number]; share: number; observed_termination_rate: number | null; count: number }[];
  probability_histogram: { lo: number; hi: number; count: number; observed_rate: number | null }[];
  global_importance: { feature: string; label: string; mean_abs_shap: number }[];
  segments: Record<"phase" | "therapeutic_area" | "funder_type" | "sites" | "sponsor_track_record" | "intervention_model", Rate[]>;
  by_start_year: { year: number; trials: number; termination_rate: number }[];
  data_quality: { steps: { step: string; removed: number }[]; excluded_for_leakage: string[] };
}

export interface SponsorHit { sponsor: string; trials: number; termination_rate: number }
