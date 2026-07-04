"""Task loading and per-fold feature encoding (shared by every model)."""

import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

from . import config as C


def load_task(name):
    path = os.path.join(C.RESULTS_DIR, f"task_{name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python cli.py preprocess` first.")
    return np.load(path, allow_pickle=True)


def full_classical(task):
    """Pool stored train/test z-scored features into one (X, y) for CV."""
    X = np.vstack([task["X_train_classical"], task["X_test_classical"]])
    y = np.concatenate([task["y_train"], task["y_test"]])
    return X, y


def encode_quantum(X_train_cl, X_test_cl):
    """PCA(N_PCA) + angle-scale to [0, pi], fit on TRAIN fold only (no leakage).

    Mirrors preprocess.build_and_save_tasks so per-fold quantum features are
    reproduced exactly from the z-scored classical features.
    """
    n_comp = min(C.N_PCA, X_train_cl.shape[1], X_train_cl.shape[0])
    pca = PCA(n_components=n_comp, random_state=C.RANDOM_STATE)
    Xtr = pca.fit_transform(X_train_cl)
    Xte = pca.transform(X_test_cl)
    lo, hi = Xtr.min(axis=0), Xtr.max(axis=0)
    denom = np.where(hi - lo == 0, 1.0, hi - lo)
    Xtr_q = np.clip(np.pi * (Xtr - lo) / denom, 0, np.pi)
    Xte_q = np.clip(np.pi * (Xte - lo) / denom, 0, np.pi)
    return Xtr_q, Xte_q


def cv_splits(y):
    """The single canonical 5-fold split reused by every model for fairness."""
    skf = StratifiedKFold(n_splits=C.N_SPLITS, shuffle=True,
                          random_state=C.RANDOM_STATE)
    return list(skf.split(np.zeros(len(y)), y))
