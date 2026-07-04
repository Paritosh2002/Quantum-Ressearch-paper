"""Importing this package registers all built-in models in the registry."""

from .base import REGISTRY, register, get_model, list_models, BaseModel  # noqa: F401
from . import classical      # noqa: F401  (registers svm_rbf / random_forest / logistic_regression)
from . import variational    # noqa: F401  (registers vqc / qnn_reupload)
from . import qsvm           # noqa: F401  (registers qsvm)
