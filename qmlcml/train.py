"""Generic training/CV runner that works for any registered model.

Handles the classical-vs-quantum feature representation, re-encodes quantum
features inside each fold (no leakage), collects out-of-fold predictions for
figures, and captures the epoch-wise history of iterative models.
"""

import numpy as np

from . import config as C
from . import data as D
from . import metrics as M
from .models import get_model


def run_model_on_task(model_name, task_name, verbose=True, **model_kwargs):
    ModelCls = get_model(model_name)
    task = D.load_task(task_name)
    X_cl, y = D.full_classical(task)
    n_classes = len(np.unique(y))
    quantum = (ModelCls.representation == "quantum")

    if verbose:
        print(f"\n== {model_name} / {task_name}: {X_cl.shape[0]} cells, "
              f"{n_classes} classes, {ModelCls.representation} features ==")

    splits = D.cv_splits(y)
    fold_metrics, histories = [], []
    oof_t, oof_p, oof_s = [], [], []

    for i, (tr, te) in enumerate(splits):
        if quantum:
            Xtr, Xte = D.encode_quantum(X_cl[tr], X_cl[te])
        else:
            Xtr, Xte = X_cl[tr], X_cl[te]
        ytr, yte = y[tr], y[te]

        model = ModelCls(n_classes, **model_kwargs)
        model.fit(Xtr, ytr, X_val=Xte, y_val=yte)
        y_pred = model.predict(Xte)
        y_score = model.predict_score(Xte)

        fold_metrics.append(M.compute_metrics(yte, y_pred, y_score, n_classes))
        oof_t.extend(yte); oof_p.extend(y_pred); oof_s.extend(np.asarray(y_score))
        if model.history() is not None:
            histories.append(model.history())
        if verbose:
            fm = fold_metrics[-1]
            print(f"  fold {i+1}/{len(splits)}  acc={fm['accuracy']:.3f} "
                  f"auc={fm['auc']:.3f}")

    extra = {"representation": ModelCls.representation}
    extra.update(model_kwargs)
    return M.save_result(
        model_name, task_name, fold_metrics,
        extra=extra,
        oof={"y_true": oof_t, "y_pred": oof_p, "y_score": oof_s},
        history=histories or None,
    )


def run(models, tasks, **model_kwargs):
    for task in tasks:
        for model in models:
            run_model_on_task(model, task, **model_kwargs)
