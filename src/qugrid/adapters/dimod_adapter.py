"""Export QuGrid problems to the D-Wave Ocean stack (``dimod``)."""

from __future__ import annotations

import numpy as np

from qugrid.problems.base import CombinatorialProblem
from qugrid.solvers.base import Result, attach_reference, finish_combinatorial


def _require_dimod():
    try:
        import dimod  # noqa: F401
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "dimod is not installed; install the extra with `pip install qugrid[dwave]`"
        ) from err


def to_bqm(problem: CombinatorialProblem):
    """``dimod.BinaryQuadraticModel`` of the problem's QUBO.

    Submit it to any Ocean sampler::

        from dwave.samplers import SimulatedAnnealingSampler
        sampleset = SimulatedAnnealingSampler().sample(to_bqm(prob), num_reads=200)
        result = result_from_sampleset(prob, sampleset)
    """
    _require_dimod()
    import dimod

    qubo = problem.qubo
    linear = {i: float(qubo.q[i, i]) for i in range(qubo.n) if qubo.q[i, i]}
    quadratic = {
        (i, j): float(2.0 * qubo.q[i, j])
        for i in range(qubo.n)
        for j in range(i + 1, qubo.n)
        if qubo.q[i, j]
    }
    return dimod.BinaryQuadraticModel(linear, quadratic, qubo.offset, vartype=dimod.BINARY)


def result_from_sampleset(problem: CombinatorialProblem, sampleset) -> Result:
    """Wrap an Ocean ``SampleSet`` into a standard QuGrid :class:`Result`."""
    _require_dimod()
    best = sampleset.first
    x = np.array([best.sample[i] for i in range(problem.qubo.n)], dtype=int)
    res = Result(solver=f"dimod:{type(sampleset).__name__}", problem=problem)
    finish_combinatorial(res, problem, x)
    res.resources["num_reads"] = len(sampleset)
    info = getattr(sampleset, "info", None)
    if info:
        timing = info.get("timing")
        if timing:
            res.resources["timing"] = timing
    return attach_reference(res, problem)
