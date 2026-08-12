"""Two-generator unit commitment over two periods, solved with QAOA.

WHAT THIS SOLVES
    Unit commitment selects which thermal units run in each period and how much
    each one produces [MW], at least total cost [$], while generation meets
    demand. This instance has 2 units and 2 periods:
        base    0 - 90 MW, cost 0.02 P^2 + 10 P + 50 $/h, startup 100 $
        peaker  0 - 60 MW, cost 0.04 P^2 + 20 P + 30 $/h, startup 80 $
        demand  60 MW in period 1, 130 MW in period 2
    The QUBO encoding uses 1 commitment bit and 2 power bits per unit and
    period, so 12 binary variables. Two power bits give the base unit the
    levels 0, 30, 60, 90 MW and the peaker the levels 0, 20, 40, 60 MW. The
    continuous optimum of this instance lies on that grid, so the encoding adds
    no error and the whole difference against the classical answer belongs to
    the solver.

WHY QUANTUM IS STUDIED HERE
    Unit commitment is a mixed-integer program, and its commitment bits map
    directly onto qubits, which makes it the most reported power system target
    for QAOA and for annealers; at 12 variables the classical answer is
    immediate, and this script measures what QAOA returns against it.

EXPECTED OUTPUT
    Four tables print: the discretization grid, the classical optimum from
    enumeration, a solver comparison in $ and MW, and the QAOA measurement
    statistics against circuit depth. Two figures are written:
      examples/figures/02_unit_commitment_qaoa_schedule.png
      examples/figures/02_unit_commitment_qaoa_solvers.png
    The script exits with status 0 when every assertion at the end holds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import qugrid as qg  # noqa: E402
from qugrid.classical.dispatch import GenParams  # noqa: E402

FIGDIR = Path(__file__).parent / "figures"
SEED = 0
DEMAND = [60.0, 130.0]
POWER_BITS = 2
QAOA_DEPTHS = (1, 2, 3)
READOUT_K = 16  # outcomes kept by a shot-limited readout

TOL_COST = 1e-6  # exact QUBO cost against enumeration [$]
TOL_GAP = 1e-9  # relative gap of the exact and annealing solvers


def bits_from_string(state: str) -> np.ndarray:
    """Convert a QuGrid top_states label (variable 0 leftmost) to a bit array."""
    return np.array([int(c) for c in state], dtype=int)


def best_of(top_states, k: int):
    """Lowest-energy outcome among the ``k`` most probable ones."""
    window = top_states[:k]
    state, _prob, _energy = min(window, key=lambda s: s[2])
    return bits_from_string(state)


def main() -> int:
    qg.viz.use_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)

    gens = [
        GenParams("base", 0, 90, 0.02, 10.0, c0=50, startup=100),
        GenParams("peaker", 0, 60, 0.04, 20.0, c0=30, startup=80),
    ]
    uc = qg.problems.UnitCommitment(gens, demand=DEMAND, power_bits=POWER_BITS)
    n = uc.n

    print(f"problem            {uc!r}")
    print(f"demand [MW]        {DEMAND}")
    print(f"\nDispatch grid with power_bits = {POWER_BITS}")
    print(f"  {'unit':>8} {'pmin':>6} {'pmax':>6} {'quantum':>9}  levels [MW]")
    for g, gen in enumerate(gens):
        levels = [float(gen.pmin + uc.delta[g] * k) for k in range(2**POWER_BITS)]
        print(f"  {gen.name:>8} {gen.pmin:>6.0f} {gen.pmax:>6.0f} {uc.delta[g]:>9.1f}  "
              f"{', '.join(f'{v:.0f}' for v in levels)}")

    # -- classical reference ----------------------------------------------
    enumerated = qg.classical.solve_uc_enumerate(gens, DEMAND)
    qubo_optimum = uc.reference()

    print("\nClassical optimum by enumeration with exact economic dispatch")
    print(f"  {'unit':>8} {'t1 on':>6} {'t1 MW':>8} {'t2 on':>6} {'t2 MW':>8}")
    for g, gen in enumerate(gens):
        print(f"  {gen.name:>8} {int(enumerated.commit[g, 0]):>6} "
              f"{enumerated.power[g, 0]:>8.1f} {int(enumerated.commit[g, 1]):>6} "
              f"{enumerated.power[g, 1]:>8.1f}")
    print(f"  total cost                {enumerated.cost:>10.4f} $")
    print(f"  QUBO ground state cost    {qubo_optimum['cost']:>10.4f} $")
    print(f"  discretization gap        {qubo_optimum['cost'] - enumerated.cost:>10.4f} $")

    # -- solvers ------------------------------------------------------------
    rows = []
    res_exact = qg.solve(uc, solver="exact", seed=SEED)
    rows.append(("exact enumeration", res_exact.decoded["cost"], res_exact.gap(),
                 res_exact.decoded["balance_error_mw"], res_exact.feasible,
                 res_exact.resources["wall_time_s"]))

    res_sa = qg.solve(uc, solver="sa", seed=SEED, n_sweeps=1500, n_restarts=8)
    rows.append(("simulated annealing", res_sa.decoded["cost"], res_sa.gap(),
                 res_sa.decoded["balance_error_mw"], res_sa.feasible,
                 res_sa.resources["wall_time_s"]))

    # QAOA is run with the full output distribution so that the measurement
    # statistics are exact. The reported solution is the best outcome among the
    # READOUT_K most probable ones, which is what a shot-limited run can see.
    qaoa_runs = []
    for p in QAOA_DEPTHS:
        res = qg.solve(uc, solver="qaoa", p=p, restarts=3, maxiter=400,
                       seed=SEED, top_k=2**n)
        x_readout = best_of(res.top_states, READOUT_K)
        decoded = uc.decode(x_readout)
        p_optimum = res.success_probability()
        qaoa_runs.append(
            {
                "p": p,
                "expectation": res.resources["expectation"],
                "p_optimum": p_optimum,
                "concentration": p_optimum * 2**n,
                "most_probable_cost": uc.decode(bits_from_string(res.top_states[0][0]))["cost"],
                "readout_cost": decoded["cost"],
                "readout_balance": decoded["balance_error_mw"],
                "readout_feasible": uc.is_feasible(x_readout),
                "wall_time_s": res.resources["wall_time_s"],
                "result": res,
                "x_readout": x_readout,
            }
        )
        rows.append((f"QAOA p={p}, best of {READOUT_K}", decoded["cost"],
                     (decoded["cost"] - enumerated.cost) / enumerated.cost,
                     decoded["balance_error_mw"], uc.is_feasible(x_readout),
                     res.resources["wall_time_s"]))

    print("\nSolver comparison, all costs in $ over the two periods")
    print(f"  {'solver':>24} {'cost [$]':>10} {'gap':>9} {'balance [MW]':>13} "
          f"{'feasible':>9} {'time [s]':>9}")
    for name, cost, gap, balance, feasible, seconds in rows:
        gap_text = "n/a" if gap is None else f"{100 * gap:>8.4f}%"
        print(f"  {name:>24} {cost:>10.4f} {gap_text:>9} {balance:>13.2f} "
              f"{'yes' if feasible else 'NO':>9} {seconds:>9.3f}")

    print("\nQAOA measurement statistics, exact statevector distribution")
    print("  <H> is the expected QUBO energy, which the optimizer minimizes. It equals")
    print("  the schedule cost only for states that meet demand; other states carry the")
    print("  balance penalty as well.")
    print(f"  {'p':>3} {'<H>':>12} {'P(optimum)':>12} {'vs uniform':>11} "
          f"{'shots for 99%':>14} {'most likely cost [$]':>21}")
    for run in qaoa_runs:
        shots = int(np.ceil(np.log(0.01) / np.log(1.0 - run["p_optimum"])))
        print(f"  {run['p']:>3} {run['expectation']:>12.1f} {run['p_optimum']:>12.5f} "
              f"{run['concentration']:>10.2f}x {shots:>14} "
              f"{run['most_probable_cost']:>21.4f}")
    print(
        f"\n  Uniform random sampling over {n} bits puts {1 / 2**n:.2e} probability on\n"
        "  the optimum. QAOA raises that, and the expectation value falls as the\n"
        "  depth grows, but the optimum still carries well under 1 percent of the\n"
        "  measurement probability. The readout matters as much as the depth: at\n"
        f"  p >= 2 the optimum is outside the {READOUT_K} most probable outcomes, so a\n"
        "  short shot budget returns a schedule that does not meet demand."
    )

    # -- figure 1: the committed schedule ---------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    qg.viz.plot_uc_schedule(res_exact.decoded, DEMAND,
                            gen_names=[g.name for g in gens], ax=ax)
    ax.set_ylim(0, max(DEMAND) * 1.30)  # headroom so the legend clears the bars
    ax.set_title(f"unit commitment schedule, QUBO optimum, {res_exact.decoded['cost']:.0f} $")
    path_schedule = FIGDIR / "02_unit_commitment_qaoa_schedule.png"
    fig.savefig(path_schedule, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # -- figure 2: solver comparison --------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    labels = [name for name, *_ in rows]
    costs = [cost for _, cost, *_ in rows]
    feasibles = [row[4] for row in rows]
    colors = [qg.viz.PALETTE[0] if ok else qg.viz.PALETTE[7] for ok in feasibles]
    axes[0].bar(range(len(costs)), costs, color=colors, edgecolor=qg.viz.SURFACE,
                linewidth=1.2)
    axes[0].axhline(enumerated.cost, color=qg.viz.INK, lw=1.4, ls=(0, (3, 3)))
    axes[0].annotate(f"classical optimum {enumerated.cost:.0f} $", (0.98, enumerated.cost),
                     xycoords=("axes fraction", "data"), ha="right", va="bottom",
                     fontsize=9, color=qg.viz.INK)
    axes[0].set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=8.5)
    axes[0].set_ylabel("total cost [$]")
    axes[0].set_ylim(0, max(costs) * 1.18)
    axes[0].set_title("cost by solver (red = demand not met)")

    depths = [run["p"] for run in qaoa_runs]
    axes[1].bar(depths, [run["p_optimum"] for run in qaoa_runs], width=0.5,
                color=qg.viz.PALETTE[0], edgecolor=qg.viz.SURFACE, linewidth=1.2)
    axes[1].axhline(1 / 2**n, color=qg.viz.INK, lw=1.4, ls=(0, (3, 3)))
    axes[1].annotate("uniform random sampling", (0.98, 1 / 2**n),
                     xycoords=("axes fraction", "data"), ha="right", va="bottom",
                     fontsize=9, color=qg.viz.INK)
    axes[1].set_xticks(depths)
    axes[1].set_xlabel("QAOA depth p")
    axes[1].set_ylabel("P(optimal bitstring)")
    axes[1].set_title("probability on the optimum")
    fig.tight_layout()
    path_solvers = FIGDIR / "02_unit_commitment_qaoa_solvers.png"
    fig.savefig(path_solvers, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nfigures            {path_schedule.name}, {path_solvers.name}")
    print(f"                   in {FIGDIR}")

    # -- self-validation --------------------------------------------------
    print("\nvalidation")
    print(f"  exact QUBO cost - enumeration cost   "
          f"{res_exact.decoded['cost'] - enumerated.cost:.3e} $")
    print(f"  simulated annealing gap              {res_sa.gap():.3e}")
    print(f"  lowest QAOA concentration vs uniform "
          f"{min(run['concentration'] for run in qaoa_runs):.2f}x")

    assert abs(res_exact.decoded["cost"] - enumerated.cost) < TOL_COST, (
        f"exact QUBO cost {res_exact.decoded['cost']} != enumeration {enumerated.cost}"
    )
    assert abs(qubo_optimum["cost"] - enumerated.cost) < TOL_COST, (
        "the discretization grid no longer contains the continuous optimum"
    )
    assert abs(res_exact.gap()) < TOL_GAP, f"exact gap {res_exact.gap()}"
    assert abs(res_sa.gap()) < TOL_GAP, f"annealing gap {res_sa.gap()}"
    assert res_sa.decoded["balance_error_mw"] < TOL_COST, "annealing does not meet demand"
    assert res_exact.feasible and res_sa.feasible, "exact or annealing answer infeasible"
    for run in qaoa_runs:
        assert run["concentration"] > 1.0, (
            f"QAOA p={run['p']} does not beat uniform sampling: "
            f"{run['concentration']:.3f}x"
        )
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
