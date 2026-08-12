"""A small quantum Boltzmann machine for renewable scenario generation.

A transverse-field Ising model at inverse temperature ``beta``:

    H = - sum_{i<j} W_ij sigma^z_i sigma^z_j - sum_i b_i sigma^z_i
        - Gamma sum_i sigma^x_i

with visible units read out in the computational basis,
``p(v) = <v| e^{-beta H} |v> / Z``. Training follows the bound-based gradient
of Amin et al., "Quantum Boltzmann Machine", PRX 2018 (arXiv:1601.02036):
for sigma^z observables the update is the familiar moment-matching rule

    dW_ij ∝ <v_i v_j>_data - <sigma^z_i sigma^z_j>_model.

Everything here is exact diagonalization — deliberately. It is trustworthy
ground truth for small models (<= ~12 units) and makes the hardware pitch
concrete: sampling *is* the bottleneck, and Ising machines / annealers /
gate-model devices enter exactly where ``2^n`` stops fitting in memory.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from qugrid.problems.scenarios import empirical_statistics


class QuantumBoltzmannMachine:
    """Visible-only transverse-field Ising Boltzmann machine (exact, small n)."""

    def __init__(self, n_visible: int, gamma: float = 1.0, beta: float = 1.0, seed: int = 0):
        if n_visible > 12:
            raise ValueError("exact-diagonalization QBM is limited to 12 units")
        self.n = int(n_visible)
        self.gamma = float(gamma)
        self.beta = float(beta)
        rng = np.random.default_rng(seed)
        self.w = 0.01 * rng.normal(size=(self.n, self.n))
        self.w = np.triu(self.w, 1)
        self.b = 0.01 * rng.normal(size=self.n)
        # spin value of each unit in each basis state (bit i of the index)
        idx = np.arange(2**self.n)
        self._z = np.array([1.0 - 2.0 * ((idx >> i) & 1) for i in range(self.n)])

    # ----------------------------------------------------------- Hamiltonian
    def _hamiltonian(self) -> np.ndarray:
        dim = 2**self.n
        diag = np.zeros(dim)
        for i in range(self.n):
            diag -= self.b[i] * self._z[i]
            for j in range(i + 1, self.n):
                if self.w[i, j] != 0.0:
                    diag -= self.w[i, j] * self._z[i] * self._z[j]
        h = np.diag(diag).astype(float)
        for i in range(self.n):
            # sigma^x_i flips bit i: pair states k <-> k ^ (1 << i)
            k = np.arange(dim)
            h[k, k ^ (1 << i)] -= self.gamma
        return h

    def state_probabilities(self) -> np.ndarray:
        """Diagonal of the Gibbs state — p(v) for every visible pattern."""
        rho = expm(-self.beta * self._hamiltonian())
        p = np.real(np.diag(rho))
        return p / p.sum()

    # -------------------------------------------------------------- sampling
    def sample(self, n_samples: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng(0)
        p = self.state_probabilities()
        draws = rng.choice(len(p), size=n_samples, p=p)
        return np.array([[(k >> i) & 1 for i in range(self.n)] for k in draws], dtype=int)

    # -------------------------------------------------------------- training
    def _model_moments(self) -> tuple[np.ndarray, np.ndarray]:
        p = self.state_probabilities()
        mean = self._z @ p  # <sigma^z_i>
        second = (self._z * p) @ self._z.T  # <sigma^z_i sigma^z_j>
        return mean, second

    def fit(
        self,
        data_bits: np.ndarray,
        epochs: int = 200,
        lr: float = 0.1,
        verbose: bool = False,
    ) -> list[float]:
        """Moment-matching training on binary data (Amin et al. bound gradient).

        Returns the trajectory of the mean absolute moment mismatch — the
        quantity the update provably decreases.
        """
        s_data = 1.0 - 2.0 * np.asarray(data_bits, dtype=float)  # bit -> spin
        mean_d = s_data.mean(axis=0)
        second_d = (s_data.T @ s_data) / len(s_data)
        errors: list[float] = []
        for epoch in range(epochs):
            mean_m, second_m = self._model_moments()
            g_b = mean_d - mean_m
            g_w = np.triu(second_d - second_m, 1)
            self.b += lr * g_b
            self.w += lr * g_w
            err = float(np.abs(g_b).mean() + np.abs(g_w).sum() / max(self.n, 1))
            errors.append(err)
            if verbose and epoch % 20 == 0:
                print(f"epoch {epoch:4d}  moment mismatch {err:.4f}")
        return errors

    # ------------------------------------------------------------- reporting
    def compare_statistics(self, data_bits: np.ndarray, n_samples: int = 2000,
                           seed: int = 1) -> dict:
        """Data vs model statistics (means, neighbor correlations)."""
        samples = self.sample(n_samples, np.random.default_rng(seed))
        return {
            "data": empirical_statistics(np.asarray(data_bits)),
            "model": empirical_statistics(samples),
        }
