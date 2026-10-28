"""Phase 3: split, imbalance handling, Optuna tuning, selection, calibration,
explainability and export."""
from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import shap
import xgboost as xgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTENC
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import config as C
from .preprocess import build_preprocessor, output_to_source_map, vif_filter, vif_table

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RISK_TIERS = [  # calibrated probability cut-offs; base rate is ~16%
    ("Low", 0.0, 0.10),
    ("Moderate", 0.10, 0.20),
    ("High", 0.20, 0.35),
    ("Critical", 0.35, 1.01),
]


def risk_tier(p: float) -> str:
    return next(name for name, lo, hi in RISK_TIERS if lo <= p < hi)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data and fold cache
# ---------------------------------------------------------------------------
@dataclass
class Fold:
    X_tr: np.ndarray
    y_tr: np.ndarray
    X_va: np.ndarray
    y_va: np.ndarray
    cat_idx: list[int]


def onehot_indices(pre) -> list[int]:
    names = pre.get_feature_names_out()
    return [i for i, n in enumerate(names) if n.startswith("cat__") or n.startswith("bin__")
            or "missingindicator" in n]


def make_folds(X: pd.DataFrame, y: np.ndarray, numeric: list[str], k: int = 5) -> list[Fold]:
    folds = []
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=C.RANDOM_STATE)
    for tr, va in skf.split(X, y):
        pre = build_preprocessor(numeric)
        Xt = pre.fit_transform(X.iloc[tr], y[tr])
        Xv = pre.transform(X.iloc[va])
        folds.append(Fold(Xt, y[tr], Xv, y[va], onehot_indices(pre)))
    return folds


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------
def make_model(name: str, params: dict, pos_weight: float):
    if name == "logreg":
        return LogisticRegression(solver="saga", l1_ratio=params["l1_ratio"],
                                  C=params["C"], class_weight="balanced" if pos_weight > 1 else None,
                                  max_iter=400, tol=1e-3)
    if name == "xgboost":
        return xgb.XGBClassifier(**params, tree_method="hist", eval_metric="aucpr",
                                 scale_pos_weight=pos_weight, n_jobs=2,
                                 random_state=C.RANDOM_STATE, verbosity=0)
    if name == "lightgbm":
        return lgb.LGBMClassifier(**params, scale_pos_weight=pos_weight, n_jobs=2,
                                  random_state=C.RANDOM_STATE, verbose=-1)
    if name == "catboost":
        return CatBoostClassifier(**params, scale_pos_weight=pos_weight, thread_count=2,
                                  random_seed=C.RANDOM_STATE, verbose=0, eval_metric="PRAUC",
                                  allow_writing_files=False)
    raise ValueError(name)


def suggest(name: str, t: optuna.Trial) -> dict:
    if name == "logreg":
        return {"C": t.suggest_float("C", 1e-3, 10, log=True),
                "l1_ratio": t.suggest_float("l1_ratio", 0.0, 1.0)}
    if name == "xgboost":
        return {"n_estimators": t.suggest_int("n_estimators", 200, 900, step=100),
                "learning_rate": t.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "max_depth": t.suggest_int("max_depth", 3, 9),
                "min_child_weight": t.suggest_float("min_child_weight", 1, 30, log=True),
                "subsample": t.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 20, log=True)}
    if name == "lightgbm":
        return {"n_estimators": t.suggest_int("n_estimators", 200, 1200, step=100),
                "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": t.suggest_int("num_leaves", 15, 255, log=True),
                "min_child_samples": t.suggest_int("min_child_samples", 10, 200, log=True),
                "subsample": t.suggest_float("subsample", 0.6, 1.0),
                "subsample_freq": 1,
                "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 20, log=True)}
    if name == "catboost":
        return {"iterations": t.suggest_int("iterations", 300, 900, step=100),
                "learning_rate": t.suggest_float("learning_rate", 0.03, 0.2, log=True),
                "depth": t.suggest_int("depth", 4, 8),
                "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1, 20, log=True)}
    raise ValueError(name)


def fit_eval(model, fold: Fold, sampling: str):
    X, y = fold.X_tr, fold.y_tr
    if sampling == "smotenc":
        X, y = SMOTENC(categorical_features=fold.cat_idx, random_state=C.RANDOM_STATE,
                       sampling_strategy=0.5).fit_resample(X, y)
    model.fit(X, y)
    p = model.predict_proba(fold.X_va)[:, 1]
    return average_precision_score(fold.y_va, p), roc_auc_score(fold.y_va, p)


def cv_score(name, params, folds, sampling, trial=None):
    pos_weight = 1.0 if sampling != "class_weight" else None
    prs, rocs = [], []
    for i, f in enumerate(folds):
        pw = pos_weight if pos_weight is not None else (len(f.y_tr) - f.y_tr.sum()) / f.y_tr.sum()
        pr, roc = fit_eval(make_model(name, params, pw), f, sampling)
        prs.append(pr)
        rocs.append(roc)
        if trial is not None:
            trial.report(float(np.mean(prs)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(prs)), float(np.mean(rocs))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(y, p, thr) -> dict:
    yhat = (p >= thr).astype(int)
    order = np.argsort(-p)
    top10 = order[: max(1, len(p) // 10)]
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "f1_macro": f1_score(y, yhat, average="macro"),
        "f1_terminated": f1_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat),
        "accuracy": accuracy_score(y, yhat),
        "brier": brier_score_loss(y, p),
        "base_rate": float(np.mean(y)),
        "top_decile_precision": float(np.mean(y[top10])),
        "top_decile_lift": float(np.mean(y[top10]) / np.mean(y)),
        "top_decile_capture": float(y[top10].sum() / y.sum()),
        "threshold": thr,
    }


def best_f1_threshold(y, p) -> float:
    grid = np.linspace(0.05, 0.6, 111)
    scores = [f1_score(y, (p >= t).astype(int), average="macro") for t in grid]
    return float(grid[int(np.argmax(scores))])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(trials: dict[str, int], timeout: int) -> None:
    feats = pd.read_parquet(C.PROCESSED_DIR / "trials_features.parquet")
    y_all = feats[C.TARGET].values

    # Stratified 70 / 15 / 15
    idx = np.arange(len(feats))
    tr_idx, tmp_idx = train_test_split(idx, test_size=0.30, stratify=y_all, random_state=C.RANDOM_STATE)
    va_idx, te_idx = train_test_split(tmp_idx, test_size=0.50, stratify=y_all[tmp_idx],
                                      random_state=C.RANDOM_STATE)
    log(f"split train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")

    # VIF filtering on train numerics (VIF > 5 removed)
    numeric_all = C.SKEWED_NUMERIC + C.MNAR_NUMERIC + C.ORDINAL_NUMERIC + C.ENGINEERED
    numeric, vif_dropped = vif_filter(feats.iloc[tr_idx], numeric_all, threshold=5.0)
    vif_final = vif_table(feats.iloc[tr_idx], numeric)
    log(f"VIF dropped: {vif_dropped}")
    features = C.LOW_CARD_CATEGORICAL + C.HIGH_CARD_CATEGORICAL + C.BINARY + numeric

    X = feats[features]
    X_tr, X_va, X_te = X.iloc[tr_idx], X.iloc[va_idx], X.iloc[te_idx]
    y_tr, y_va, y_te = y_all[tr_idx], y_all[va_idx], y_all[te_idx]

    log("building 5-fold preprocessing cache")
    folds = make_folds(X_tr, y_tr, numeric)

    # Imbalance strategy experiment on a fixed LightGBM
    base = {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63,
            "min_child_samples": 40, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.7}
    imbalance = {}
    for s in ["none", "class_weight", "smotenc"]:
        pr, roc = cv_score("lightgbm", base, folds, s)
        imbalance[s] = {"cv_pr_auc": pr, "cv_roc_auc": roc}
        log(f"imbalance {s}: PR-AUC={pr:.4f} ROC-AUC={roc:.4f}")
    sampling = max(imbalance, key=lambda k: imbalance[k]["cv_pr_auc"])
    log(f"chosen imbalance strategy: {sampling}")

    # Optuna per model
    studies = {}
    for name, n in trials.items():
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=C.RANDOM_STATE),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=1))
        model_sampling = "class_weight" if name == "logreg" else sampling

        def objective(t, name=name, model_sampling=model_sampling):
            pr, roc = cv_score(name, suggest(name, t), folds, model_sampling, trial=t)
            t.set_user_attr("cv_roc_auc", roc)
            return pr

        t0 = time.time()
        study.optimize(objective, n_trials=n, timeout=timeout)
        bt = study.best_trial
        studies[name] = {"best_params": bt.params, "cv_pr_auc": bt.value,
                         "cv_roc_auc": bt.user_attrs["cv_roc_auc"], "trials": len(study.trials),
                         "minutes": round((time.time() - t0) / 60, 1), "sampling": model_sampling}
        log(f"{name}: CV PR-AUC={bt.value:.4f} ROC-AUC={bt.user_attrs['cv_roc_auc']:.4f} "
            f"({len(study.trials)} trials)")

    # Refit every tuned model on full train, compare on validation
    pre = build_preprocessor(numeric)
    Xt = pre.fit_transform(X_tr, y_tr)
    Xv, Xs = pre.transform(X_va), pre.transform(X_te)
    cat_idx = onehot_indices(pre)
    pw_full = (len(y_tr) - y_tr.sum()) / y_tr.sum()
    fitted = {}
    for name, info in studies.items():
        s = info["sampling"]
        params = dict(info["best_params"])
        if name == "lightgbm":
            params["subsample_freq"] = 1
        model = make_model(name, params, pw_full if s == "class_weight" else 1.0)
        Xf, yf = (SMOTENC(categorical_features=cat_idx, random_state=C.RANDOM_STATE,
                          sampling_strategy=0.5).fit_resample(Xt, y_tr) if s == "smotenc" else (Xt, y_tr))
        model.fit(Xf, yf)
        pv = model.predict_proba(Xv)[:, 1]
        info["val_pr_auc"] = average_precision_score(y_va, pv)
        info["val_roc_auc"] = roc_auc_score(y_va, pv)
        fitted[name] = model
        log(f"{name}: val PR-AUC={info['val_pr_auc']:.4f} ROC-AUC={info['val_roc_auc']:.4f}")

    best_name = max(studies, key=lambda k: studies[k]["val_pr_auc"])
    best = fitted[best_name]
    log(f"selected model: {best_name}")

    # Calibrate on validation (sigmoid: smooth, so small design changes move the score), choose threshold on validation, report on test
    calibrator = CalibratedClassifierCV(FrozenEstimator(best), method="sigmoid")
    calibrator.fit(Xv, y_va)
    p_va = calibrator.predict_proba(Xv)[:, 1]
    thr = best_f1_threshold(y_va, p_va)
    p_te = calibrator.predict_proba(Xs)[:, 1]
    test_metrics = metrics(y_te, p_te, thr)
    log(f"TEST: {json.dumps({k: round(v, 4) for k, v in test_metrics.items()})}")
    tiers = pd.Series([risk_tier(p) for p in p_te]).value_counts()
    tier_stats = []
    for name, lo, hi in RISK_TIERS:
        mask = np.array([risk_tier(p) == name for p in p_te])
        tier_stats.append({"tier": name, "range": [lo, min(hi, 1.0)], "share": float(mask.mean()),
                           "observed_termination_rate": float(y_te[mask].mean()) if mask.any() else None,
                           "count": int(tiers.get(name, 0))})

    # Temporal robustness: train on early trials, test on later ones
    yrs = feats["start_year"].values
    early = np.where(yrs <= 2014)[0]
    late = np.where(yrs >= 2017)[0]
    pre_t = build_preprocessor(numeric)
    Xe = pre_t.fit_transform(X.iloc[early], y_all[early])
    m_t = clone(best)
    m_t.fit(Xe, y_all[early])
    p_late = m_t.predict_proba(pre_t.transform(X.iloc[late]))[:, 1]
    temporal = {"train": "start 2000-2014", "test": "start 2017-2021",
                "n_train": int(len(early)), "n_test": int(len(late)),
                "roc_auc": roc_auc_score(y_all[late], p_late),
                "pr_auc": average_precision_score(y_all[late], p_late),
                "test_base_rate": float(y_all[late].mean())}
    log(f"temporal check: {temporal}")

    # SHAP (global) on a test sample
    src_map = output_to_source_map(pre, features)
    rng = np.random.default_rng(C.RANDOM_STATE)
    sample = rng.choice(len(Xs), size=min(3000, len(Xs)), replace=False)
    if best_name == "logreg":
        bg = Xt[rng.choice(len(Xt), 2000, replace=False)]
        np.save(C.ARTIFACTS_DIR / "logreg_background.npy", bg)
        explainer = shap.LinearExplainer(best, bg)
    else:
        explainer = shap.TreeExplainer(best)
    sv = explainer.shap_values(Xs[sample])
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    agg = pd.DataFrame(sv, columns=src_map).T.groupby(level=0).sum().T
    global_imp = agg.abs().mean().sort_values(ascending=False)
    np.save(C.ARTIFACTS_DIR / "shap_test_sample.npy", agg.values)
    pd.DataFrame(X_te.iloc[sample].values, columns=features).to_parquet(
        C.ARTIFACTS_DIR / "shap_test_features.parquet", index=False)
    (C.ARTIFACTS_DIR / "shap_columns.json").write_text(json.dumps(list(agg.columns)))

    # Export
    expected = explainer.expected_value
    expected = float(np.ravel(expected)[-1])
    bundle = {"preprocessor": pre, "model": best, "calibrator": calibrator, "model_name": best_name,
              "features": features, "numeric": numeric, "source_map": src_map,
              "threshold": thr, "risk_tiers": RISK_TIERS, "shap_expected_value": expected,
              "train_medians": X_tr[numeric].median().to_dict()}
    joblib.dump(bundle, C.ARTIFACTS_DIR / "model_bundle.joblib", compress=3)

    meta = {
        "dataset": {"rows": len(feats), "train": len(tr_idx), "val": len(va_idx), "test": len(te_idx),
                    "termination_rate": float(y_all.mean()),
                    "start_year_window": [C.START_YEAR_MIN, C.START_YEAR_MAX]},
        "vif_dropped": vif_dropped,
        "vif_final": vif_final.round(2).to_dict(orient="records"),
        "imbalance_experiment": imbalance, "imbalance_strategy": sampling,
        "models": studies, "selected_model": best_name,
        "test_metrics": test_metrics, "risk_tiers": tier_stats, "temporal_validation": temporal,
        "test_probability_histogram": [
            {"lo": float(lo), "hi": float(hi), "count": int(((p_te >= lo) & (p_te < hi)).sum()),
             "observed_rate": float(y_te[(p_te >= lo) & (p_te < hi)].mean())
             if ((p_te >= lo) & (p_te < hi)).any() else None}
            for lo, hi in zip(np.arange(0, 1.0, 0.025), np.arange(0.025, 1.025, 0.025))],
        "global_shap": [{"feature": k, "label": C.FEATURE_LABELS.get(k, k), "mean_abs_shap": float(v)}
                        for k, v in global_imp.items()],
    }
    (C.ARTIFACTS_DIR / "model_metadata.json").write_text(json.dumps(meta, indent=2, default=float))
    log("saved artifacts")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="few trials, for smoke tests")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per model study")
    a = ap.parse_args()
    trials = ({"logreg": 2, "xgboost": 2, "lightgbm": 2, "catboost": 1} if a.quick
              else {"logreg": 12, "xgboost": 25, "lightgbm": 30, "catboost": 10})
    run(trials, a.timeout)
