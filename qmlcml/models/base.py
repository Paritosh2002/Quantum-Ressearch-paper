"""Common model interface + a registry so new algorithms are one small file.

To add an algorithm, subclass BaseModel, set `name`/`representation`, implement
fit/predict/predict_score, and decorate the class with @register. It then shows
up in `python cli.py list-models` and is trainable via `--model <name>`.
"""

import numpy as np

REGISTRY = {}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


def get_model(name):
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]


def list_models():
    return sorted(REGISTRY.items(), key=lambda kv: (REGISTRY[kv[0]].representation, kv[0]))


class BaseModel:
    name = "base"
    representation = "classical"    # "classical" -> z-scored 77 genes; "quantum" -> PCA8 angles
    iterative = False               # True if it produces an epoch-wise training history

    def __init__(self, n_classes, **kwargs):
        self.n_classes = n_classes
        self.kwargs = kwargs

    # X_val/y_val let iterative models log a validation curve; others ignore them.
    def fit(self, X, y, X_val=None, y_val=None):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_score(self, X):
        """1-D positive-class score (binary) or (n, n_classes) (multiclass)."""
        raise NotImplementedError

    def history(self):
        """Iterative models return {'train_loss':[...], 'train_acc':[...],
        'val_acc':[...]}; others return None."""
        return None


def scores_to_binary_or_multi(proba, n_classes):
    return proba[:, 1] if n_classes == 2 else proba
