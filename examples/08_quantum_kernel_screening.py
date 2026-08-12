"""Quantum kernel bandwidth study for N-1 security screening of the WSCC 9-bus system.

WHAT
    An operating point is insecure when a single branch outage pushes some
    branch loading above its rating. Operators screen thousands of such points,
    so a fast classifier is a useful surrogate for the full contingency
    analysis. This script samples 160 load patterns of the WSCC 9-bus system,
    labels each one by exact DC power flow over all 6 non-bridge single-branch
    outages, and trains a classifier on the fidelity quantum kernel of
    Havlicek et al. (Nature 2019, arXiv:1804.11326).

WHY QUANTUM
    The quantum kernel has one dominant hyperparameter: the angle range that
    the features are scaled into, which acts as the kernel bandwidth
    (Shaydulin and Wild, Phys. Rev. A 2022, arXiv:2111.05451). This script
    measures how much of the outcome that single choice decides, and compares
    every setting against an RBF kernel with the median heuristic. No quantum
    advantage is claimed. The best quantum setting ties the classical
    baseline; the wrong setting is no better than a coin toss.

EXPECTED OUTPUT
    A 3 x 2 table of test accuracy over angle ranges (0.25 pi, 0.5 pi, 1.0 pi)
    and feature-map repetitions (1, 2). Test accuracy falls from 1.000 at
    0.25 pi with 1 repetition to 0.500 at 1.0 pi with 2 repetitions, while
    training accuracy stays at or above 0.866. Alignment between the kernel
    matrix and the labels falls in the same order, from 0.116 to 0.045. The
    RBF baseline holds 1.000 test accuracy and an alignment of 0.186, above
    every quantum setting. Two figures are written to examples/figures/.
    Runtime is about 1 second.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import qugrid as qg  # noqa: E402
from qugrid.solvers import (  # noqa: E402
    compare_kernels,
    quantum_kernel,
    rbf_kernel,
    scale_features,
)
from qugrid.viz import PALETTE, use_style  # noqa: E402

N_SAMPLES = 160
THRESHOLD = 1.0
DATA_SEED = 1
SPLIT_SEED = 0
BANDWIDTHS = (0.25, 0.5, 1.0)  # multiples of pi
REPETITIONS = (1, 2)
FIGDIR = Path(__file__).parent / "figures"


def target_alignment(k: np.ndarray, y: np.ndarray) -> float:
    """Cosine similarity between the kernel matrix and the ideal label kernel.

    With labels mapped to -1 and +1, the ideal kernel is ``t t^T``: pairs in
    the same class have value 1, pairs in different classes -1. A higher value
    means the kernel geometry carries more of the label structure, which is
    what a kernel machine can use.
    """
    t = 2.0 * np.asarray(y, dtype=float) - 1.0
    ideal = np.outer(t, t)
    return float((k * ideal).sum() / (np.linalg.norm(k) * np.linalg.norm(ideal)))


def split_scaled(data, hi: float):
    """Scale every feature into ``[0, hi]``, then split into train and test."""
    scaled = qg.problems.ScreeningDataset(
        x=scale_features(data.x, lo=0.0, hi=hi),
        y=data.y,
        feature_names=data.feature_names,
        threshold=data.threshold,
        contingencies=data.contingencies,
        net_name=data.net_name,
    )
    return scaled.split(seed=SPLIT_SEED)


def main() -> int:
    use_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)

    net = qg.cases.case9()
    data = qg.problems.screening_dataset(
        net, n_samples=N_SAMPLES, threshold=THRESHOLD, seed=DATA_SEED
    )
    insecure = int(data.y.sum())
    print(f"network            {net.name}, {net.n_bus} buses, {net.n_branch} branches")
    print(f"features           {len(data.feature_names)} load multipliers: "
          f"{', '.join(data.feature_names)}")
    print(f"contingencies      {len(data.contingencies)} non-bridge branch outages "
          f"{data.contingencies}")
    print(f"label              insecure if max loading > {data.threshold:.2f} "
          f"in any screened outage")
    print(f"class balance      {insecure} insecure / {len(data.y) - insecure} secure "
          f"of {len(data.y)} samples")

    # The load multipliers are sampled from a known design range, so min-max
    # scaling to the angle range uses no label information.
    print(f"qubits             {data.x.shape[1]} (one per feature)\n")

    # The RBF kernel with the median heuristic is invariant to a common
    # rescaling of all features, so one baseline covers every angle range.
    x_base, y_base, _, _ = split_scaled(data, hi=1.0)
    rbf_alignment = target_alignment(rbf_kernel(x_base), y_base)

    rows = []
    for factor in BANDWIDTHS:
        x_train, y_train, x_test, y_test = split_scaled(data, hi=factor * np.pi)
        for reps in REPETITIONS:
            out = compare_kernels(x_train, y_train, x_test, y_test, reps=reps)
            rows.append(
                {
                    "factor": factor,
                    "reps": reps,
                    "q_train": out["quantum"]["train_accuracy"],
                    "q_test": out["quantum"]["test_accuracy"],
                    "rbf_train": out["rbf"]["train_accuracy"],
                    "rbf_test": out["rbf"]["test_accuracy"],
                    "alignment": target_alignment(
                        quantum_kernel(x_train, reps=reps), y_train
                    ),
                }
            )

    header = (
        f"{'angle range':>12}  {'reps':>4}  {'quantum train':>14}  {'quantum test':>13}  "
        f"{'RBF test':>9}  {'alignment':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['factor']:>10.2f} pi  {r['reps']:>4}  {r['q_train']:>14.3f}  "
            f"{r['q_test']:>13.3f}  {r['rbf_test']:>9.3f}  {r['alignment']:>10.4f}"
        )

    best = max(rows, key=lambda r: r["q_test"])
    worst = min(rows, key=lambda r: r["q_test"])
    print(f"\nbest quantum setting   {best['factor']:.2f} pi, {best['reps']} rep(s): "
          f"test accuracy {best['q_test']:.3f}")
    print(f"worst quantum setting  {worst['factor']:.2f} pi, {worst['reps']} rep(s): "
          f"test accuracy {worst['q_test']:.3f} "
          f"(training accuracy {worst['q_train']:.3f})")
    print(f"RBF baseline           test accuracy {rows[0]['rbf_test']:.3f}, "
          f"no tuning beyond the median heuristic")
    print(f"\nalignment with the labels: RBF {rbf_alignment:.4f}, best quantum "
          f"{best['alignment']:.4f}, worst quantum {worst['alignment']:.4f}.")
    print("Wider angles lower the alignment, and test accuracy follows it down")
    print("while training accuracy stays high.")

    # ---------------------------------------------------------------- figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.9))
    for i, reps in enumerate(REPETITIONS):
        subset = [r for r in rows if r["reps"] == reps]
        ax1.plot(
            [r["factor"] for r in subset],
            [r["q_test"] for r in subset],
            marker="o",
            color=PALETTE[i],
            label=f"quantum, {reps} rep(s)",
        )
        ax2.plot(
            [r["factor"] for r in subset],
            [r["alignment"] for r in subset],
            marker="o",
            color=PALETTE[i],
            label=f"quantum, {reps} rep(s)",
        )
    ax1.axhline(rows[0]["rbf_test"], color=PALETTE[7], lw=1.4, ls=(0, (3, 3)))
    ax1.annotate(
        f"RBF baseline {rows[0]['rbf_test']:.2f}",
        (BANDWIDTHS[-1], rows[0]["rbf_test"]),
        xytext=(0, -14),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=PALETTE[7],
    )
    ax1.axhline(0.5, color="#898781", lw=1.0, ls=(0, (1, 3)))
    ax1.annotate(
        "coin toss",
        (BANDWIDTHS[0], 0.5),
        xytext=(0, 6),
        textcoords="offset points",
        fontsize=9,
        color="#898781",
    )
    ax1.set_xticks(list(BANDWIDTHS))
    ax1.set_xlabel("angle range / pi (kernel bandwidth)")
    ax1.set_ylabel("test accuracy")
    ax1.set_ylim(0.4, 1.05)
    ax1.set_title("Bandwidth decides the quantum kernel")
    ax1.legend(fontsize=9, loc="lower left")
    ax2.axhline(rbf_alignment, color=PALETTE[7], lw=1.4, ls=(0, (3, 3)))
    ax2.annotate(
        f"RBF baseline {rbf_alignment:.3f}",
        (BANDWIDTHS[-1], rbf_alignment),
        xytext=(0, -14),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=PALETTE[7],
    )
    ax2.set_xticks(list(BANDWIDTHS))
    ax2.set_xlabel("angle range / pi (kernel bandwidth)")
    ax2.set_ylabel("alignment with the labels")
    ax2.set_title("Wide angles hide the label structure")
    ax2.legend(fontsize=9)
    fig.suptitle(f"Quantum kernel for N-1 screening, {net.name}, {len(data.y)} samples")
    fig.tight_layout()
    fig.savefig(FIGDIR / "08_quantum_kernel_screening_bandwidth.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(BANDWIDTHS), figsize=(11.0, 3.4))
    scaled_sets = [scale_features(data.x, lo=0.0, hi=f * np.pi) for f in BANDWIDTHS]
    for ax, factor, xs in zip(axes, BANDWIDTHS, scaled_sets):
        gram = quantum_kernel(xs[:40], reps=1)
        im = ax.imshow(gram, cmap=qg.viz.sequential_cmap(), vmin=0.0, vmax=1.0)
        ax.set_title(f"{factor:.2f} pi")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("kernel value")
    cbar.outline.set_visible(False)
    fig.suptitle("Quantum kernel matrix of 40 operating points, 1 repetition")
    fig.savefig(FIGDIR / "08_quantum_kernel_screening_gram.png", dpi=150)
    plt.close(fig)
    print(f"\nfigures -> {FIGDIR}/08_quantum_kernel_screening_*.png")

    # ------------------------------------------------------------- validation
    assert 0.25 <= data.y.mean() <= 0.75, f"degenerate class balance: {data.y.mean():.3f}"
    assert best["q_test"] > 0.8, f"best quantum test accuracy is only {best['q_test']:.3f}"
    assert rows[0]["rbf_test"] > 0.8, f"RBF baseline is only {rows[0]['rbf_test']:.3f}"
    assert best["q_test"] - worst["q_test"] > 0.2, (
        "bandwidth had almost no effect, so the study claim does not hold: "
        f"{worst['q_test']:.3f} to {best['q_test']:.3f}"
    )
    print("VALIDATION PASSED: the best quantum setting and the RBF baseline both")
    print("exceed 0.8 test accuracy, and bandwidth moves the quantum result by")
    print(f"{100 * (best['q_test'] - worst['q_test']):.0f} accuracy points.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
