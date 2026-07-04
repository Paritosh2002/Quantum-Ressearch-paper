"""Metric computation and per-(model, task) result serialization."""

import os
import json
import glob
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score,
)

from . import config as C


def compute_metrics(y_true, y_pred, y_score, n_classes):
    """y_score: 1-D positive-class score (binary) or (n, n_classes) (multiclass)."""
    out = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1_macro":  f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro":    recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    try:
        if n_classes == 2:
            out["auc"] = roc_auc_score(y_true, y_score)
        else:
            out["auc"] = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
    except ValueError:
        out["auc"] = float("nan")
    return out


def _safe(reduce, vals):
    vals = [v for v in vals if v == v]   # drop NaN
    return float(reduce(vals)) if vals else float("nan")


def save_result(model_name, task_name, fold_metrics, extra=None, oof=None, history=None):
    """Write results/metrics/<model>__<task>.json.

    oof:     dict with concatenated out-of-fold y_true/y_pred/y_score (for figures)
    history: list (per fold) of {epoch metric -> [values]} for iterative models
    """
    os.makedirs(C.METRICS_DIR, exist_ok=True)
    keys = fold_metrics[0].keys()
    summary = {k: _safe(np.mean, [fm[k] for fm in fold_metrics]) for k in keys}
    stds    = {k: _safe(np.std,  [fm[k] for fm in fold_metrics]) for k in keys}
    payload = {
        "model": model_name, "task": task_name, "n_folds": len(fold_metrics),
        "per_fold": fold_metrics, "mean": summary, "std": stds,
    }
    if extra:
        payload.update(extra)
    if oof:
        payload["oof"] = {k: np.asarray(oof[k]).tolist()
                          for k in ("y_true", "y_pred", "y_score")}
    if history:
        payload["history"] = history
    path = os.path.join(C.METRICS_DIR, f"{model_name}__{task_name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    m = summary
    print(f"  [{model_name} / {task_name}]  acc={m['accuracy']:.3f}  "
          f"auc={m['auc']:.3f}  f1={m['f1_macro']:.3f}  -> {os.path.relpath(path, C.ROOT)}")
    return payload


def load_results():
    rows = []
    for path in sorted(glob.glob(os.path.join(C.METRICS_DIR, "*.json"))):
        with open(path) as f:
            rows.append(json.load(f))
    return rows
