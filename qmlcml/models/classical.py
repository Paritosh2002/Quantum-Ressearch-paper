"""Classical baselines on the z-scored full-feature (77-gene) representation.

All use class_weight="balanced" because nBM is a strong minority (~173 vs 1168 Dx).
"""

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .. import config as C
from .base import BaseModel, register, scores_to_binary_or_multi


class _SklearnModel(BaseModel):
    representation = "classical"

    def _make(self):
        raise NotImplementedError

    def fit(self, X, y, X_val=None, y_val=None):
        self.clf = self._make()
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_score(self, X):
        proba = self.clf.predict_proba(X)
        return scores_to_binary_or_multi(proba, self.n_classes)


@register
class SVMRBF(_SklearnModel):
    name = "svm_rbf"
    def _make(self):
        return SVC(kernel="rbf", C=1.0, class_weight="balanced",
                   probability=True, random_state=C.RANDOM_STATE)


@register
class RandomForest(_SklearnModel):
    name = "random_forest"
    def _make(self):
        return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                      random_state=C.RANDOM_STATE, n_jobs=-1)


@register
class LogReg(_SklearnModel):
    name = "logistic_regression"
    def _make(self):
        return LogisticRegression(max_iter=2000, class_weight="balanced",
                                  random_state=C.RANDOM_STATE)
