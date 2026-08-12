"""Quantum kernel methods for grid classification tasks.

Implements the fidelity quantum kernel of Havlíček et al., Nature 2019
(arXiv:1804.11326): embed a feature vector into a quantum state with the
ZZ feature map, then use ``K(x, y) = |<phi(x)|phi(y)>|^2`` inside any kernel
machine. QuGrid pairs it with a dependency-free kernel ridge classifier and
an RBF classical baseline, because the research question is never "does the
quantum kernel work" but "does it beat a tuned classical kernel here".

States are computed exactly (features become qubits, so keep dimensions
small — exactly the regime of :func:`qugrid.problems.screening_dataset`).
"""

from __future__ import annotations

import numpy as np

from qugrid.solvers.statevector import apply_h_all, diag_phase, zero_state


def scale_features(x: np.ndarray, lo: float = 0.0, hi: float = np.pi / 2) -> np.ndarray:
    """Min-max scale features into ``[lo, hi]`` (column-wise), as the feature
    map expects angles.

    The angle range is the quantum kernel's *bandwidth* — its single most
    important hyperparameter (Shaydulin & Wild, Phys. Rev. A 2022,
    arXiv:2111.05451). Too wide and the kernel matrix collapses toward the
    identity (perfect training fit, chance-level generalization); the
    ``pi/2`` default is a robust starting point at these feature counts.
    """
    x = np.asarray(x, dtype=float)
    xmin, xmax = x.min(axis=0), x.max(axis=0)
    span = np.where(xmax > xmin, xmax - xmin, 1.0)
    return lo + (hi - lo) * (x - xmin) / span


def _zz_feature_state(x: np.ndarray, reps: int = 1) -> np.ndarray:
    """ZZ feature map |phi(x)> on ``len(x)`` qubits (exact statevector)."""
    d = len(x)
    idx = np.arange(2**d)
    z = np.array([1.0 - 2.0 * ((idx >> i) & 1) for i in range(d)])  # (d, 2^d) of +-1
    phase = np.zeros(2**d)
    for i in range(d):
        phase += x[i] * z[i]
    for i in range(d):
        for j in range(i + 1, d):
            phase += (np.pi - x[i]) * (np.pi - x[j]) * z[i] * z[j]
    psi = zero_state(d)
    for _ in range(reps):
        apply_h_all(psi, d)
        diag_phase(psi, phase)
    return psi


def quantum_kernel(x1: np.ndarray, x2: np.ndarray | None = None, reps: int = 1) -> np.ndarray:
    """Fidelity kernel matrix ``K_ij = |<phi(x1_i)|phi(x2_j)>|^2``.

    ``reps=1`` by default: deeper maps sharpen the kernel and, combined with a
    wide angle range, push it toward the identity matrix. Increase ``reps``
    only together with a narrower :func:`scale_features` range.
    """
    x1 = np.asarray(x1, dtype=float)
    states1 = np.array([_zz_feature_state(row, reps) for row in x1])
    if x2 is None:
        states2 = states1
    else:
        states2 = np.array([_zz_feature_state(row, reps) for row in np.asarray(x2, float)])
    overlaps = states1.conj() @ states2.T
    return np.abs(overlaps) ** 2


def rbf_kernel(x1: np.ndarray, x2: np.ndarray | None = None, gamma: float | None = None):
    """Classical RBF baseline; ``gamma`` defaults to the median heuristic."""
    x1 = np.asarray(x1, dtype=float)
    x2 = x1 if x2 is None else np.asarray(x2, dtype=float)
    d2 = ((x1[:, None, :] - x2[None, :, :]) ** 2).sum(-1)
    if gamma is None:
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        gamma = 1.0 / max(med, 1e-12)
    return np.exp(-gamma * d2)


class KernelRidgeClassifier:
    """Binary classifier on a precomputed kernel (labels in {0, 1}).

    Kernel ridge regression on ±1 targets, thresholded at zero — 15 lines,
    no scikit-learn, adequate for benchmark-scale studies. Swap in any kernel
    machine you like; the kernel matrix is the quantum part.
    """

    def __init__(self, ridge: float = 1e-3):
        self.ridge = float(ridge)
        self._alpha: np.ndarray | None = None

    def fit(self, k_train: np.ndarray, y: np.ndarray) -> KernelRidgeClassifier:
        t = 2.0 * np.asarray(y, dtype=float) - 1.0
        n = k_train.shape[0]
        self._alpha = np.linalg.solve(k_train + self.ridge * np.eye(n), t)
        return self

    def decision(self, k_x_train: np.ndarray) -> np.ndarray:
        assert self._alpha is not None, "call fit() first"
        return k_x_train @ self._alpha

    def predict(self, k_x_train: np.ndarray) -> np.ndarray:
        return (self.decision(k_x_train) > 0).astype(int)


def compare_kernels(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    reps: int = 1,
    ridge: float = 1e-3,
) -> dict:
    """Train the same classifier on the quantum and RBF kernels; report both.

    Features must already be scaled (see :func:`scale_features`).
    """
    kernels = {
        "quantum": (
            quantum_kernel(x_train, reps=reps),
            quantum_kernel(x_test, x_train, reps=reps),
        ),
        "rbf": (rbf_kernel(x_train), rbf_kernel(x_test, x_train)),
    }
    out: dict = {}
    for name, (k_tr, k_te) in kernels.items():
        clf = KernelRidgeClassifier(ridge=ridge).fit(k_tr, y_train)
        out[name] = {
            "train_accuracy": float((clf.predict(k_tr) == y_train).mean()),
            "test_accuracy": float((clf.predict(k_te) == y_test).mean()),
        }
    out["n_qubits"] = x_train.shape[1]
    return out
