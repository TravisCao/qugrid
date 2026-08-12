"""Quantum Boltzmann machine trained on wind scenarios.

WHAT
    Stochastic unit commitment and reserve studies need scenarios: sample
    trajectories that reproduce the joint statistics of the real resource,
    not just its average. This script trains a 5-unit quantum Boltzmann
    machine on 300 binarized wind profiles over a 5-period horizon, then
    checks whether the trained model reproduces the two statistics that
    matter for scenario use: the mean output per period, and the correlation
    between neighboring periods, which controls ramps.

WHY QUANTUM
    The model is a transverse-field Ising model, and training follows the
    bound-based gradient of Amin et al. (Phys. Rev. X 2018, arXiv:1601.02036).
    Here it is solved by exact diagonalization of the 32-state Hamiltonian, so
    the numbers are ground truth rather than an approximation. That also shows
    where hardware would enter: the cost is the partition function, which
    doubles with every added period, and quantum annealers or gate-model
    samplers are proposed exactly for that step. Nothing here runs faster than
    a classical Boltzmann machine of the same size.

EXPECTED OUTPUT
    A training curve for the moment mismatch, which falls from 1.355 to 0.017
    over 150 epochs, a factor of 82. A comparison table of data and model
    statistics: the mean output per period agrees within 0.013, and the
    neighbor correlation within 0.014. Two figures are written to
    examples/figures/. Runtime is about 1 second.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from qugrid.problems import (  # noqa: E402
    binarize,
    empirical_statistics,
    toy_wind_profiles,
)
from qugrid.solvers import QuantumBoltzmannMachine  # noqa: E402
from qugrid.viz import PALETTE, use_style  # noqa: E402

N_PROFILES = 300
HORIZON = 5
GAMMA = 0.8
EPOCHS = 150
LEARNING_RATE = 0.15
N_MODEL_SAMPLES = 4000
SEED = 0
FIGDIR = Path(__file__).parent / "figures"


def main() -> int:
    use_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)

    profiles = toy_wind_profiles(n_profiles=N_PROFILES, horizon=HORIZON, seed=SEED)
    bits = binarize(profiles)
    threshold = float(np.median(profiles))
    data_stats = empirical_statistics(bits)

    print(f"data               {N_PROFILES} wind profiles over {HORIZON} periods")
    print(f"binarization       capacity factor > {threshold:.3f} (median), "
          f"{100 * bits.mean():.1f}% of bits are high")
    print(f"model              transverse-field Ising, {HORIZON} visible units, "
          f"gamma = {GAMMA}")
    print(f"state space        {2 ** HORIZON} states, diagonalized exactly")
    print(f"training           {EPOCHS} epochs, learning rate {LEARNING_RATE}\n")

    qbm = QuantumBoltzmannMachine(n_visible=HORIZON, gamma=GAMMA, seed=SEED)
    errors = qbm.fit(bits, epochs=EPOCHS, lr=LEARNING_RATE)
    stats = qbm.compare_statistics(bits, n_samples=N_MODEL_SAMPLES, seed=1)

    data_means = np.asarray(data_stats["means"])
    data_corr = np.asarray(data_stats["neighbor_corr"])
    model_means = np.asarray(stats["model"]["means"])
    model_corr = np.asarray(stats["model"]["neighbor_corr"])
    mean_error = float(np.abs(data_means - model_means).max())
    corr_error = float(np.abs(data_corr - model_corr).max())

    print(f"moment mismatch    {errors[0]:.4f} at epoch 0 -> {errors[-1]:.4f} at "
          f"epoch {EPOCHS} (factor {errors[0] / errors[-1]:.1f})\n")

    header = f"{'period':>7}  {'data mean':>10}  {'model mean':>11}  {'difference':>11}"
    print(header)
    print("-" * len(header))
    for t in range(HORIZON):
        print(
            f"{t + 1:>7}  {data_means[t]:>10.4f}  {model_means[t]:>11.4f}  "
            f"{model_means[t] - data_means[t]:>+11.4f}"
        )
    print()
    header = f"{'periods':>7}  {'data corr':>10}  {'model corr':>11}  {'difference':>11}"
    print(header)
    print("-" * len(header))
    for t in range(HORIZON - 1):
        print(
            f"{t + 1}-{t + 2:<5}  {data_corr[t]:>10.4f}  {model_corr[t]:>11.4f}  "
            f"{model_corr[t] - data_corr[t]:>+11.4f}"
        )
    print(f"\nlargest mean error        {mean_error:.4f}")
    print(f"largest correlation error {corr_error:.4f}")
    print(f"model samples drawn       {N_MODEL_SAMPLES}")

    # ---------------------------------------------------------------- figures
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(errors, color=PALETTE[0])
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean absolute moment mismatch")
    ax.set_title(f"Moment matching converges in {EPOCHS} epochs")
    ax.annotate(
        f"{errors[0]:.3f} -> {errors[-1]:.3f}",
        (0.98, 0.92),
        xycoords="axes fraction",
        ha="right",
        fontsize=9,
        color=PALETTE[0],
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "09_qbm_wind_scenarios_training.png", dpi=150)
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.9))
    width = 0.38
    t_mean = np.arange(HORIZON)
    ax1.bar(t_mean - width / 2, data_means, width, label="data", color=PALETTE[0],
            edgecolor="white", linewidth=0.8)
    ax1.bar(t_mean + width / 2, model_means, width, label="model", color=PALETTE[1],
            edgecolor="white", linewidth=0.8)
    ax1.set_xticks(t_mean, [f"t{t + 1}" for t in range(HORIZON)])
    ax1.set_xlabel("period")
    ax1.set_ylabel("mean high-output rate")
    ax1.set_ylim(0.0, 0.75)
    ax1.set_title(f"Means agree within {mean_error:.3f}")
    ax1.legend(fontsize=9)

    t_corr = np.arange(HORIZON - 1)
    ax2.bar(t_corr - width / 2, data_corr, width, label="data", color=PALETTE[0],
            edgecolor="white", linewidth=0.8)
    ax2.bar(t_corr + width / 2, model_corr, width, label="model", color=PALETTE[1],
            edgecolor="white", linewidth=0.8)
    ax2.set_xticks(t_corr, [f"t{t + 1}-t{t + 2}" for t in range(HORIZON - 1)])
    ax2.set_xlabel("period pair")
    ax2.set_ylabel("neighbor correlation")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_title(f"Ramp structure agrees within {corr_error:.3f}")
    ax2.legend(fontsize=9)
    fig.suptitle(
        f"Quantum Boltzmann machine, {N_PROFILES} wind profiles, "
        f"{N_MODEL_SAMPLES} model samples"
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "09_qbm_wind_scenarios_statistics.png", dpi=150)
    plt.close(fig)
    print(f"\nfigures -> {FIGDIR}/09_qbm_wind_scenarios_*.png")

    # ------------------------------------------------------------- validation
    assert errors[-1] < 0.25 * errors[0], (
        f"training did not reduce the moment mismatch enough: "
        f"{errors[0]:.4f} -> {errors[-1]:.4f}"
    )
    assert mean_error < 0.08, f"model means are off by {mean_error:.4f}"
    assert corr_error < 0.15, f"neighbor correlations are off by {corr_error:.4f}"
    assert data_corr.min() > 0.5, (
        f"the data has no ramp structure to learn: lowest neighbor correlation "
        f"is {data_corr.min():.3f}"
    )
    print(f"VALIDATION PASSED: the moment mismatch fell to "
          f"{100 * errors[-1] / errors[0]:.1f}% of its initial value, the means agree")
    print(f"within {mean_error:.3f}, and the neighbor correlations within {corr_error:.3f}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
