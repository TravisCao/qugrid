"""Solve QuGrid problems *through* the external stacks, same :class:`Result`.

The export adapters hand a problem to another library and stop there. These
functions take the round trip: build the vendor object, run the vendor
solver, decode the answer back into engineering units, and return the
standard :class:`~qugrid.solvers.base.Result` — feasibility of the original
constraints, gap against the classical reference computed in the same run,
and the resource bill with wall time. They are registered as ordinary
:func:`qugrid.solve` names:

* ``"dimod-exact"`` — ``dimod.ExactSolver``, the Ocean twin of the built-in
  ``"exact"`` enumeration.
* ``"dwave-sa"`` — ``dwave.samplers.SimulatedAnnealingSampler``, the mature
  reference for the built-in ``"sa"``.
* ``"qiskit-qaoa"`` — ``MinimumEigenOptimizer`` from qiskit-optimization
  around Qiskit's QAOA on the statevector sampler primitive.

Every import stays lazy, so the core install needs no quantum SDK, and a
missing extra raises the same message the export adapters raise.
"""

from __future__ import annotations

import numpy as np

from qugrid.problems.base import CombinatorialProblem
from qugrid.solvers.base import Result, attach_reference, finish_combinatorial, timed

#: Exhaustive solvers stop here, the limit the internal reference uses.
MAX_EXACT_VARS = 24


def _require(module: str, extra: str) -> None:
    import importlib

    try:
        importlib.import_module(module)
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            f"{module} is not installed; install the extra with "
            f"`pip install qugrid[{extra}]`"
        ) from err


def _bits_from_sample(problem: CombinatorialProblem, sample) -> np.ndarray:
    """Ocean sample (variable index -> 0/1) as a QuGrid bit array."""
    return np.array([sample[i] for i in range(problem.qubo.n)], dtype=int)


def solve_dimod_exact(problem: CombinatorialProblem, **_ignored) -> Result:
    """Exhaustive enumeration with ``dimod.ExactSolver`` (n <= 24)."""
    _require("dimod", "dwave")
    import dimod

    from qugrid.adapters.dimod_adapter import to_bqm

    n = problem.qubo.n
    if n > MAX_EXACT_VARS:
        raise ValueError(f"refusing to enumerate 2^{n} states")
    res = Result(solver="dimod-exact", problem=problem)
    with timed(res.resources):
        sampleset = dimod.ExactSolver().sample(to_bqm(problem))
        x = _bits_from_sample(problem, sampleset.first.sample)
    finish_combinatorial(res, problem, x)
    res.resources.update({"n_vars": n, "states_enumerated": len(sampleset)})
    return attach_reference(res, problem)


def solve_dwave_sa(
    problem: CombinatorialProblem,
    num_reads: int = 100,
    num_sweeps: int = 1000,
    seed: int = 0,
    **_ignored,
) -> Result:
    """Simulated annealing with ``dwave.samplers.SimulatedAnnealingSampler``.

    The Ocean stack's production annealer (C++ inner loop), run on the same
    QUBO as the built-in ``"sa"`` so the two are directly comparable.
    """
    _require("dwave.samplers", "dwave")
    from dwave.samplers import SimulatedAnnealingSampler

    from qugrid.adapters.dimod_adapter import to_bqm

    res = Result(solver="dwave-sa", problem=problem)
    with timed(res.resources):
        sampleset = SimulatedAnnealingSampler().sample(
            to_bqm(problem), num_reads=num_reads, num_sweeps=num_sweeps, seed=seed
        )
        x = _bits_from_sample(problem, sampleset.first.sample)
    finish_combinatorial(res, problem, x)
    res.resources.update(
        {
            "n_vars": problem.qubo.n,
            "num_reads": num_reads,
            "num_sweeps": num_sweeps,
            "seed": seed,
        }
    )
    return attach_reference(res, problem)


def _quadratic_program(problem: CombinatorialProblem):
    """The problem's QUBO as a qiskit-optimization ``QuadraticProgram``."""
    from qiskit_optimization import QuadraticProgram

    qubo = problem.qubo
    qp = QuadraticProgram("qugrid")
    for i in range(qubo.n):
        # Index names, not qubo.names: formulations use characters (@, [, ])
        # that qiskit-optimization's model exporters reject.
        qp.binary_var(f"x{i}")
    qp.minimize(
        constant=float(qubo.offset),
        linear={i: float(qubo.q[i, i]) for i in range(qubo.n)},
        quadratic={
            (i, j): float(2.0 * qubo.q[i, j])
            for i in range(qubo.n)
            for j in range(i + 1, qubo.n)
        },
    )
    return qp


def solve_qiskit_qaoa(
    problem: CombinatorialProblem,
    reps: int = 2,
    maxiter: int = 200,
    seed: int = 0,
    **_ignored,
) -> Result:
    """QAOA through qiskit-optimization's ``MinimumEigenOptimizer``.

    Shot-based sampling on Qiskit's statevector primitive, COBYLA outer
    loop. Slower than the built-in ``"qaoa"`` by two orders of magnitude at
    the same depth — the point is that a second implementation of the same
    algorithm lands on the same objective.
    """
    _require("qiskit_optimization", "qiskit")
    _require("qiskit_algorithms", "qiskit")
    from qiskit.primitives import StatevectorSampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_algorithms.utils import algorithm_globals
    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    algorithm_globals.random_seed = seed  # fixes QAOA's random initial angles
    evaluations: list[int] = []
    res = Result(solver="qiskit-qaoa", problem=problem)
    with timed(res.resources):
        # The QAOA ansatz carries a PauliEvolutionGate. Synthesizing it once
        # through a pass manager, rather than at every objective call inside
        # the sampler, is worth two orders of magnitude in wall time.
        passes = generate_preset_pass_manager(
            optimization_level=1, basis_gates=["h", "rx", "rz", "cx"]
        )
        qaoa = QAOA(
            StatevectorSampler(seed=seed),
            COBYLA(maxiter=maxiter),
            reps=reps,
            transpiler=passes,
            callback=lambda count, *_rest: evaluations.append(count),
        )
        out = MinimumEigenOptimizer(qaoa).solve(_quadratic_program(problem))
        x = np.asarray(out.x, dtype=int)
    finish_combinatorial(res, problem, x)
    res.resources.update(
        {
            "n_qubits": problem.qubo.n,
            "p": reps,
            "evaluations": evaluations[-1] if evaluations else 0,
            "seed": seed,
        }
    )
    return attach_reference(res, problem)
