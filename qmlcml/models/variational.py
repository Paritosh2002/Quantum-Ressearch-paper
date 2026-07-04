"""Variational quantum classifiers (PennyLane + PyTorch).

A shared training loop (VariationalModel) records a per-epoch history of
train loss / train accuracy / validation accuracy so training and testing can be
visualized. Concrete algorithms only differ in their circuit:

  * VQC  -- AngleEmbedding(RY) + StronglyEntanglingLayers
  * QNN  -- data re-uploading + BasicEntanglerLayers (a distinct ansatz family)

Add another variational algorithm by subclassing VariationalModel and overriding
`_build_qnode` (and `_weight_shapes`).
"""

import numpy as np
import torch
import pennylane as qml

from .. import config as C
from .base import BaseModel, register

torch.manual_seed(C.RANDOM_STATE)


class VariationalModel(BaseModel):
    representation = "quantum"
    iterative = True

    n_layers = 3
    epochs   = 60
    lr       = 0.05

    def __init__(self, n_classes, **kwargs):
        super().__init__(n_classes, **kwargs)
        self.n_qubits = C.N_PCA
        self.n_out = 1 if n_classes == 2 else n_classes
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.qnode = self._build_qnode()
        self._init_params()
        self._hist = {"train_loss": [], "train_acc": [], "val_acc": []}

    # --- to override in subclasses ------------------------------------------
    def _build_qnode(self):
        raise NotImplementedError

    def _init_params(self):
        raise NotImplementedError

    def _circuit_out(self, x):
        """Return (batch, n_out) pre-bias logits from the qnode."""
        outs = self.qnode(x, *self._params())
        return torch.stack(outs, dim=-1)

    def _params(self):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

    # --- shared training loop -----------------------------------------------
    def _forward(self, x):
        return self._circuit_out(x) + self.bias

    def _class_weights(self, y):
        counts = np.bincount(y, minlength=self.n_classes).astype(float)
        counts[counts == 0] = 1.0
        return counts.sum() / (self.n_classes * counts)

    def fit(self, X, y, X_val=None, y_val=None):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        Xt = torch.tensor(X, dtype=torch.float64)
        yt = torch.tensor(y, dtype=torch.long)
        w = torch.tensor(self._class_weights(y), dtype=torch.float64)

        if self.n_classes == 2:
            pos_weight = torch.tensor([w[1] / w[0]], dtype=torch.float64)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            target = yt.double().unsqueeze(1)
        else:
            loss_fn = torch.nn.CrossEntropyLoss(weight=w)

        for _ in range(self.epochs):
            opt.zero_grad()
            logits = self._forward(Xt)
            loss = loss_fn(logits, target) if self.n_classes == 2 else loss_fn(logits, yt)
            loss.backward()
            opt.step()

            self._hist["train_loss"].append(float(loss.item()))
            self._hist["train_acc"].append(self._accuracy(logits.detach(), y))
            if X_val is not None:
                with torch.no_grad():
                    vlogits = self._forward(torch.tensor(X_val, dtype=torch.float64))
                self._hist["val_acc"].append(self._accuracy(vlogits, y_val))
        return self

    def _accuracy(self, logits, y_true):
        pred = self._logits_to_pred(logits)
        return float((pred == np.asarray(y_true)).mean())

    def _logits_to_pred(self, logits):
        if self.n_classes == 2:
            return (torch.sigmoid(logits).squeeze(1).numpy() >= 0.5).astype(int)
        return torch.softmax(logits, dim=1).numpy().argmax(axis=1)

    def predict(self, X):
        with torch.no_grad():
            logits = self._forward(torch.tensor(X, dtype=torch.float64))
        return self._logits_to_pred(logits)

    def predict_score(self, X):
        with torch.no_grad():
            logits = self._forward(torch.tensor(X, dtype=torch.float64))
        if self.n_classes == 2:
            return torch.sigmoid(logits).squeeze(1).numpy()
        return torch.softmax(logits, dim=1).numpy()

    def history(self):
        return self._hist


@register
class VQC(VariationalModel):
    name = "vqc"
    n_layers = 3

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_out)]
        return circuit

    def _init_params(self):
        shape = qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers,
                                                   n_wires=self.n_qubits)
        self.weights = torch.nn.Parameter(0.1 * torch.randn(*shape, dtype=torch.float64))
        self.bias = torch.nn.Parameter(torch.zeros(self.n_out, dtype=torch.float64))

    def _params(self):       return (self.weights,)
    def parameters(self):    return [self.weights, self.bias]


@register
class QNNReupload(VariationalModel):
    """Data re-uploading classifier: alternate encoding + entangling blocks."""
    name = "qnn_reupload"
    n_layers = 2          # re-uploading blocks
    epochs   = 60

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            for L in range(self.n_layers):
                qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation="Y")
                qml.BasicEntanglerLayers(weights[L], wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_out)]
        return circuit

    def _init_params(self):
        # BasicEntanglerLayers wants (blocks_of_1, n_wires); we stack n_layers re-uploads
        shape = (self.n_layers, 1, self.n_qubits)
        self.weights = torch.nn.Parameter(0.1 * torch.randn(*shape, dtype=torch.float64))
        self.bias = torch.nn.Parameter(torch.zeros(self.n_out, dtype=torch.float64))

    def _params(self):       return (self.weights,)
    def parameters(self):    return [self.weights, self.bias]
