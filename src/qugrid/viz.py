"""Publication-quality plots with zero configuration.

Every figure QuGrid produces follows one visual system: a colorblind-safe
categorical palette, one-hue sequential ramps, thin marks, recessive grid.
Call :func:`use_style` once (solvers' plot helpers do it for you) and every
matplotlib figure in your session inherits the look.
"""

from __future__ import annotations

import numpy as np

# Categorical palette (colorblind-safe, fixed order — never cycle further).
PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
SEQ_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV_STEPS = ["#0d366b", "#3987e5", "#9ec5f4", "#f0efec", "#f0a5a5", "#e34948", "#8f1d1d"]


def _mpl():
    import matplotlib
    import matplotlib.pyplot as plt

    return matplotlib, plt


def sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("qugrid_seq", SEQ_STEPS)


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("qugrid_div", DIV_STEPS)


def use_style() -> None:
    """Apply the QuGrid matplotlib style globally (idempotent)."""
    matplotlib, _ = _mpl()
    from cycler import cycler

    matplotlib.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.prop_cycle": cycler(color=PALETTE),
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


# --------------------------------------------------------------------- layouts
_HAND_LAYOUTS: dict[str, dict[int, tuple[float, float]]] = {
    "toy3": {1: (0.0, 0.0), 2: (2.0, 0.0), 3: (1.0, -1.5)},
    "case5": {1: (0.0, 0.0), 2: (1.6, 0.0), 3: (3.2, 0.0), 4: (2.4, -1.4), 5: (0.8, -1.4)},
    "case9": {
        1: (0.0, -2.6),
        4: (0.0, -1.3),
        5: (-1.15, -0.65),
        6: (1.15, -0.65),
        9: (-1.15, 0.65),
        7: (1.15, 0.65),
        8: (0.0, 1.3),
        2: (0.0, 2.6),
        3: (2.3, 1.3),
    },
    "case14": {
        1: (0.0, 1.0),
        2: (1.2, 0.0),
        3: (3.8, 0.0),
        4: (3.2, 1.2),
        5: (1.9, 1.35),
        6: (1.6, 2.9),
        7: (3.7, 2.2),
        8: (4.7, 2.2),
        9: (3.7, 3.1),
        10: (3.1, 3.8),
        11: (2.2, 3.6),
        12: (0.7, 3.9),
        13: (1.9, 4.4),
        14: (3.3, 4.6),
    },
}


def layout_positions(net) -> dict[int, tuple[float, float]]:
    """Node positions: hand-drawn one-line layouts for bundled small cases,
    spectral layout with light force refinement otherwise."""
    if net.name in _HAND_LAYOUTS:
        return {net.bus_index(b): xy for b, xy in _HAND_LAYOUTS[net.name].items()}
    a = net.adjacency()
    deg = np.diag(a.sum(axis=1))
    lap = deg - a
    _, vecs = np.linalg.eigh(lap)
    pos = vecs[:, 1:3].copy()
    span = np.abs(pos).max() or 1.0
    pos = pos / span
    # a few rounds of repulsion to open up collapsed clusters
    rng = np.random.default_rng(0)
    pos += 0.01 * rng.normal(size=pos.shape)
    for _ in range(60):
        diff = pos[:, None, :] - pos[None, :, :]
        dist2 = (diff**2).sum(-1) + 1e-6
        np.fill_diagonal(dist2, np.inf)
        repulse = (diff / dist2[:, :, None]).sum(axis=1) * 0.004
        attract = np.zeros_like(pos)
        for f, t, _w in net.edges():
            d = pos[f] - pos[t]
            attract[f] -= 0.06 * d
            attract[t] += 0.06 * d
        pos += repulse + attract
    return {i: (float(pos[i, 0]), float(pos[i, 1])) for i in range(net.n_bus)}


# ----------------------------------------------------------------------- plots
def plot_network(
    net,
    node_values: np.ndarray | None = None,
    islands: np.ndarray | None = None,
    highlight_buses: list[int] | None = None,
    cut_edges: list[tuple[int, int]] | None = None,
    edge_loading: np.ndarray | None = None,
    node_label: str = "",
    ax=None,
    title: str | None = None,
):
    """One-line-diagram style network plot.

    ``node_values`` colors buses on the sequential ramp (with colorbar);
    ``islands`` colors buses categorically; ``highlight_buses`` (internal
    indices) draws filled markers (for example PMU locations); ``cut_edges``
    draws those branches dashed; ``edge_loading`` (0-1 per branch) widens and
    darkens loaded branches.
    """
    use_style()
    _, plt = _mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 5.4))
    pos = layout_positions(net)
    cut_set = {frozenset(e) for e in (cut_edges or [])}

    edges = [(f, t) for f, t, _ in net.edges()]
    for line, (f, t) in enumerate(edges):
        x = [pos[f][0], pos[t][0]]
        y = [pos[f][1], pos[t][1]]
        if frozenset((f, t)) in cut_set:
            ax.plot(x, y, ls=(0, (4, 3)), color=PALETTE[7], lw=2.2, zorder=1)
            continue
        lw, color = 1.6, BASELINE
        if edge_loading is not None and np.isfinite(edge_loading[line]):
            lw = 1.0 + 3.5 * float(np.clip(edge_loading[line], 0, 1.3))
            color = sequential_cmap()(0.15 + 0.85 * float(np.clip(edge_loading[line], 0, 1)))
        ax.plot(x, y, color=color, lw=lw, zorder=1, solid_capstyle="round")

    xs = np.array([pos[i][0] for i in range(net.n_bus)])
    ys = np.array([pos[i][1] for i in range(net.n_bus)])
    gen_buses = set(net.gen_bus[net.gen_on].tolist())

    if islands is not None:
        colors = [PALETTE[int(v) % len(PALETTE)] for v in islands]
        sc = ax.scatter(xs, ys, s=340, c=colors, edgecolors=SURFACE, linewidths=2, zorder=3)
    elif node_values is not None:
        sc = ax.scatter(
            xs, ys, s=340, c=node_values, cmap=sequential_cmap(),
            edgecolors=SURFACE, linewidths=2, zorder=3,
        )
        cb = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
        cb.set_label(node_label, color=INK_2)
        cb.outline.set_visible(False)
    else:
        face = [PALETTE[0] if i in gen_buses else "#ffffff" for i in range(net.n_bus)]
        sc = ax.scatter(
            xs, ys, s=340, c=face, edgecolors=BASELINE, linewidths=1.4, zorder=3
        )
    del sc

    if highlight_buses:
        hx = [pos[i][0] for i in highlight_buses]
        hy = [pos[i][1] for i in highlight_buses]
        ax.scatter(hx, hy, s=560, facecolors="none", edgecolors=PALETTE[1],
                   linewidths=2.6, zorder=4)

    for i in range(net.n_bus):
        dark_face = islands is not None or node_values is not None
        ax.annotate(
            str(int(net.bus[i, 0])),
            pos[i],
            ha="center",
            va="center",
            fontsize=8.5,
            color=SURFACE if dark_face else INK_2,
            zorder=5,
            fontweight="bold",
        )
    for i in sorted(gen_buses):
        ax.annotate(
            "G",
            (pos[i][0], pos[i][1]),
            xytext=(0, 16),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=MUTED,
            zorder=5,
        )

    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax


def plot_convergence(results, ax=None, log: bool = True, reference: float | None = None):
    """Optimizer trajectories for one result or a dict/list of results."""
    use_style()
    _, plt = _mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 3.8))
    if not isinstance(results, (list, dict)):
        results = {getattr(results, "solver", "run"): results}
    if isinstance(results, list):
        results = {f"{r.solver} #{i}": r for i, r in enumerate(results)}
    ref = reference
    for idx, (label, res) in enumerate(results.items()):
        if not res.history:
            continue
        best = np.minimum.accumulate(np.asarray(res.history, dtype=float))
        ax.plot(best, label=label, color=PALETTE[idx % len(PALETTE)])
        if ref is None and res.reference:
            ref = res.reference.get("objective")
    if ref is not None:
        ax.axhline(ref, color=MUTED, lw=1.2, ls=(0, (3, 3)))
        ax.annotate("classical optimum", (0.99, ref), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=8.5, color=MUTED)
    if log:
        finite = [v for r in results.values() for v in r.history if np.isfinite(v)]
        if ref is not None and finite and min(finite) > ref:
            ax.set_yscale("linear")
    ax.set_xlabel("objective evaluations")
    ax.set_ylabel("best objective so far")
    if len(results) > 1:
        ax.legend(loc="upper right", fontsize=9)
    return ax


def plot_uc_schedule(decoded: dict, demand, gen_names: list[str] | None = None, ax=None):
    """Unit commitment result: dispatch stack against demand, commitment strip."""
    use_style()
    _, plt = _mpl()
    power = np.asarray(decoded["power"], dtype=float)
    n_gen, n_t = power.shape
    names = gen_names or [f"unit {g + 1}" for g in range(n_gen)]
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 3.9))
    t = np.arange(n_t)
    bottom = np.zeros(n_t)
    for g in range(n_gen):
        ax.bar(t, power[g], bottom=bottom, width=0.62, label=names[g],
               color=PALETTE[g % len(PALETTE)], edgecolor=SURFACE, linewidth=1.2)
        bottom += power[g]
    ax.step(
        np.concatenate([[-0.5], t + 0.5]),
        np.concatenate([[demand[0]], np.asarray(demand, float)]),
        where="pre", color=INK, lw=2.0, label="demand",
    )
    ax.set_xticks(t, [f"t{k + 1}" for k in range(n_t)])
    ax.set_xlabel("period")
    ax.set_ylabel("power [MW]")
    ax.legend(loc="upper left", fontsize=9, ncols=min(n_gen + 1, 4))
    return ax


def plot_qubo(qubo, ax=None, title: str | None = None):
    """The QUBO coefficient matrix on a diverging scale — formulation X-ray."""
    use_style()
    _, plt = _mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(5.4, 4.6))
    q = qubo.q
    vmax = float(np.abs(q).max()) or 1.0
    im = ax.imshow(q, cmap=diverging_cmap(), vmin=-vmax, vmax=vmax)
    cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.outline.set_visible(False)
    cb.set_label("coefficient", color=INK_2)
    ax.set_xlabel("variable index")
    ax.set_ylabel("variable index")
    ax.grid(False)
    if title:
        ax.set_title(title)
    return ax


def plot_distribution(result, top: int = 12, ax=None):
    """Output distribution of a quantum solver, optimum highlighted."""
    use_style()
    _, plt = _mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 3.6))
    states = result.top_states[:top]
    if not states:
        raise ValueError("result has no sampled states (classical solver?)")
    labels = [s for s, _, _ in states]
    probs = [p for _, p, _ in states]
    energies = np.array([e for _, _, e in states])
    ref = (result.reference or {}).get("objective")
    tol = 1e-9 * max(1.0, abs(ref)) if ref is not None else 0.0
    colors = [
        PALETTE[0] if ref is not None and abs(e - ref) <= tol else BASELINE
        for e in energies
    ]
    ax.bar(range(len(probs)), probs, color=colors, edgecolor=SURFACE, linewidth=1.0)
    ax.set_xticks(range(len(labels)), labels, rotation=60, fontsize=7.5, family="monospace")
    ax.set_ylabel("probability")
    ax.set_xlabel("bitstring (variable 0 leftmost)")
    if ref is not None:
        ax.annotate("blue = optimal states", (0.99, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=9, color=PALETTE[0])
    return ax


def plot_scenarios(profiles: np.ndarray, samples: np.ndarray | None = None, ax=None):
    """Wind scenario fans: historical data vs generated samples."""
    use_style()
    _, plt = _mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 3.8))
    t = np.arange(profiles.shape[1])
    for row in profiles[:60]:
        ax.plot(t, row, color=PALETTE[0], alpha=0.10, lw=1.0)
    ax.plot(t, profiles.mean(axis=0), color=PALETTE[0], lw=2.4, label="data mean")
    if samples is not None:
        for row in samples[:60]:
            ax.plot(t, row, color=PALETTE[1], alpha=0.10, lw=1.0)
        ax.plot(t, samples.mean(axis=0), color=PALETTE[1], lw=2.4, ls="--",
                label="model mean")
    ax.set_xlabel("period")
    ax.set_ylabel("capacity factor")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", fontsize=9)
    return ax
