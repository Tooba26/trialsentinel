"""Export the selected gradient-boosted model to ONNX and check it matches the Python model.

The preprocessing pipeline (MICE imputation, target encoding) stays in Python; the ONNX graph takes the
already-transformed feature matrix. This lets the model run in onnxruntime (for example, a lighter serving
image or an edge deployment) while keeping one source of truth for feature engineering.

    PYTHONPATH=src python scripts/export_onnx.py
"""
from __future__ import annotations

import json
import sys

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

from trialsentinel import config as C


def main() -> None:
    b = joblib.load(C.ARTIFACTS_DIR / "model_bundle.joblib")
    model, name = b["model"], b["model_name"]
    df = pd.read_parquet(C.PROCESSED_DIR / "trials_features.parquet").sample(500, random_state=0)
    X = b["preprocessor"].transform(df[b["features"]]).astype(np.float32)
    n = X.shape[1]

    from onnxmltools.convert.common.data_types import FloatTensorType
    if name == "lightgbm":
        from onnxmltools import convert_lightgbm as convert
        onx = convert(model, initial_types=[("input", FloatTensorType([None, n]))], target_opset=15,
                      zipmap=False)
    elif name == "xgboost":
        from onnxmltools import convert_xgboost as convert
        booster = model.get_booster()
        booster.feature_names = None  # ONNX converter expects f0..fn names
        onx = convert(model, initial_types=[("input", FloatTensorType([None, n]))], target_opset=15)
    elif name == "catboost":
        path = C.ARTIFACTS_DIR / "model.onnx"
        model.save_model(str(path), format="onnx")
        onx = None
    else:
        from skl2onnx import to_onnx
        onx = to_onnx(model, X[:1], target_opset=15, options={id(model): {"zipmap": False}})

    path = C.ARTIFACTS_DIR / "model.onnx"
    if onx is not None:
        path.write_bytes(onx.SerializeToString())

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {sess.get_inputs()[0].name: X})
    if isinstance(outs[-1], list):  # zipmap output: list of {class: prob}
        probs = np.array([[d[0], d[1]] for d in outs[-1]])
    else:
        probs = next(o for o in outs if isinstance(o, np.ndarray) and o.ndim == 2)
    diff = float(np.max(np.abs(probs[:, 1] - model.predict_proba(X)[:, 1])))
    report = {"model": name, "path": str(path.relative_to(C.ROOT)), "inputs": n,
              "max_abs_diff_vs_python": diff, "size_kb": round(path.stat().st_size / 1024, 1)}
    print(json.dumps(report, indent=2))
    if diff > 1e-4:
        sys.exit("ONNX output differs from the Python model")


if __name__ == "__main__":
    main()
