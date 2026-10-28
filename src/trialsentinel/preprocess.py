"""Preprocessing pipeline: imputation, encoding and scaling."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder, RobustScaler,
                                   StandardScaler, TargetEncoder)
from statsmodels.stats.outliers_influence import variance_inflation_factor

from . import config as C


def vif_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = df[cols].astype(float)
    X = X.fillna(X.median())
    X = (X - X.mean()) / X.std().replace(0, 1)
    X.insert(0, "const", 1.0)
    vals = [variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])]
    return pd.DataFrame({"feature": cols, "vif": vals}).sort_values("vif", ascending=False)


def vif_filter(df: pd.DataFrame, cols: list[str], threshold: float = 5.0,
               protect: tuple[str, ...] = ()) -> tuple[list[str], list[dict]]:
    """Iteratively drop the highest-VIF numeric feature until all VIF <= threshold.
    Composite (engineered) features are dropped before raw ones when tied."""
    keep = list(cols)
    dropped = []
    while True:
        tbl = vif_table(df, keep)
        tbl = tbl[~tbl["feature"].isin(protect)]
        if tbl.empty or tbl["vif"].iloc[0] <= threshold:
            return keep, dropped
        over = tbl[tbl["vif"] > threshold]
        composites = over[over["feature"].isin(C.ENGINEERED)]
        worst = composites.iloc[0] if len(composites) else over.iloc[0]
        dropped.append({"feature": worst["feature"], "vif": round(float(worst["vif"]), 2)})
        keep.remove(worst["feature"])


def build_preprocessor(numeric_cols: list[str]) -> ColumnTransformer:
    skewed = [c for c in C.SKEWED_NUMERIC if c in numeric_cols]
    rest = [c for c in C.MNAR_NUMERIC + C.ORDINAL_NUMERIC + C.ENGINEERED if c in numeric_cols]
    return ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50,
                                         sparse_output=False)),
            ]), C.LOW_CARD_CATEGORICAL),
            ("te", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("target", TargetEncoder(target_type="binary", smooth="auto", cv=5,
                                         random_state=C.RANDOM_STATE)),
            ]), C.HIGH_CARD_CATEGORICAL),
            ("bin", "passthrough", C.BINARY),
            ("skew", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("scale", RobustScaler()),
            ]), skewed),
            ("mice", Pipeline([
                ("impute", IterativeImputer(max_iter=10, random_state=C.RANDOM_STATE,
                                            add_indicator=True, skip_complete=True)),
                ("scale", StandardScaler()),
            ]), rest),
        ],
        verbose_feature_names_out=True,
    )


def output_to_source_map(pre: ColumnTransformer, input_cols: list[str]) -> list[str]:
    """Map every transformed column back to the user-facing feature it came from."""
    names = pre.get_feature_names_out()
    by_len = sorted(input_cols, key=len, reverse=True)
    out = []
    for n in names:
        body = n.split("__", 1)[1]
        body = body.removeprefix("missingindicator_")
        src = next((c for c in by_len if body == c or body.startswith(c + "_")), body)
        out.append(src)
    return out
