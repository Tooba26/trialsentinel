import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const pct = (v: number, digits = 0) => `${(v * 100).toFixed(digits)}%`;

export const TIER_COLOR: Record<string, string> = {
  Low: "var(--risk-low)",
  Moderate: "var(--risk-mid)",
  High: "var(--risk-high)",
  Critical: "var(--risk-critical)",
};

export const MODEL_NAMES: Record<string, string> = {
  logreg: "Logistic regression (ElasticNet)",
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
  catboost: "CatBoost",
};

export function humanize(code: string) {
  const map: Record<string, string> = {
    PHASE2: "Phase 2", PHASE3: "Phase 3", "PHASE1|PHASE2": "Phase 1/2", "PHASE2|PHASE3": "Phase 2/3",
    INDUSTRY: "Industry", OTHER: "Other", NIH: "NIH", NETWORK: "Network", OTHER_GOV: "Other government",
    FED: "US federal", INDIV: "Individual", UNKNOWN: "Unknown",
    PARALLEL: "Parallel", SINGLE_GROUP: "Single group", CROSSOVER: "Crossover",
    FACTORIAL: "Factorial", SEQUENTIAL: "Sequential",
  };
  return map[code] ?? code.charAt(0).toUpperCase() + code.slice(1).toLowerCase().replace(/_/g, " ");
}
