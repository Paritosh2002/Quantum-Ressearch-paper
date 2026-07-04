"""Quantum-Kernel SVM: ZZ feature map -> statevector fidelity kernel -> SVC.

The quantum kernel is the state fidelity  K(x, x') = |<phi(x)|phi(x')>|^2  of the
ZZFeatureMap embedding. Qiskit's FidelityQuantumKernel computes this with ONE
circuit per pair -> O(n^2) circuit executions, which is intractable on a CPU
simulator (an 80x80 kernel took >2 min here).

Instead we exploit exact statevector simulation: compute each sample's embedded
statevector ONCE (O(n) circuit builds), then the whole Gram matrix is a single
matrix product  |S S^H|^2. This is mathematically identical to the fidelity kernel
for pure-state embeddings but runs in seconds, so no subsampling is needed.
"""

import numpy as np
from sklearn.svm import SVC
from qiskit.quantum_info import Statevector

from .. import config as C
from .base import BaseModel, register, scores_to_binary_or_multi

try:   # Qiskit >= 2.1 functional API
    from qiskit.circuit.library import zz_feature_map as _zz
    def _make_feature_map(dim, reps):
        return _zz(feature_dimension=dim, reps=reps)
except ImportError:
    from qiskit.circuit.library import ZZFeatureMap
    def _make_feature_map(dim, reps):
        return ZZFeatureMap(feature_dimension=dim, reps=reps)


@register
class QSVM(BaseModel):
    name = "qsvm"
    representation = "quantum"

    reps = 2
    max_train = None      # statevector kernel is fast; use all data by default

    def __init__(self, n_classes, **kwargs):
        super().__init__(n_classes, **kwargs)
        self.max_train = kwargs.get("max_train", self.max_train)
        self.reps = kwargs.get("reps", self.reps)
        self.fmap = _make_feature_map(C.N_PCA, self.reps)
        self.params = list(self.fmap.parameters)

    # --- embedding: one statevector per sample -------------------------------
    def _statevectors(self, X):
        S = np.empty((len(X), 2 ** C.N_PCA), dtype=complex)
        for i, x in enumerate(X):
            bound = self.fmap.assign_parameters(dict(zip(self.params, x)))
            S[i] = Statevector(bound).data
        return S

    @staticmethod
    def _kernel(S_a, S_b):
        # |<a|b>|^2 for every pair -> real Gram matrix in [0, 1]
        return np.abs(S_a @ S_b.conj().T) ** 2

    def _subsample(self, X, y):
        if self.max_train is None or len(y) <= self.max_train:
            return X, y
        rng = np.random.RandomState(C.RANDOM_STATE)
        classes, counts = np.unique(y, return_counts=True)
        alloc = np.maximum(2, np.round(self.max_train * counts / counts.sum()).astype(int))
        idx = []
        for c, n_c in zip(classes, alloc):
            c_idx = np.where(y == c)[0]
            idx.extend(rng.choice(c_idx, size=min(n_c, len(c_idx)), replace=False))
        return X[np.array(idx)], y[np.array(idx)]

    def fit(self, X, y, X_val=None, y_val=None):
        self.X_train, self.y_train = self._subsample(X, y)
        self.S_train = self._statevectors(self.X_train)
        K = self._kernel(self.S_train, self.S_train)
        self.svc = SVC(kernel="precomputed", class_weight="balanced",
                       probability=True, random_state=C.RANDOM_STATE)
        self.svc.fit(K, self.y_train)
        return self

    def _kernel_to_train(self, X):
        # cache: the runner calls predict() then predict_score() on the same X
        if getattr(self, "_cache_id", None) != id(X):
            self._cache_id, self._cache_S = id(X), self._statevectors(X)
        return self._kernel(self._cache_S, self.S_train)

    def predict(self, X):
        return self.svc.predict(self._kernel_to_train(X))

    def predict_score(self, X):
        proba = self.svc.predict_proba(self._kernel_to_train(X))
        return scores_to_binary_or_multi(proba, self.n_classes)
