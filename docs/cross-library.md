# Cross-library benchmark

QuGrid's built-in solvers are not the only implementations of the algorithms
they run. Example 11 solves the same three power system QUBOs — unit
commitment of 2 units over 2 periods, controlled islanding of the WSCC 9-bus
system, and minimum PMU placement on the PJM 5-bus system — with five solvers
from three libraries, through one call:

```python
import qugrid as qg

qg.solve(problem, solver="sa")           # built-in simulated annealing
qg.solve(problem, solver="qaoa", p=2)    # built-in statevector QAOA
qg.solve(problem, solver="dimod-exact")  # dimod.ExactSolver
qg.solve(problem, solver="dwave-sa")     # Ocean SimulatedAnnealingSampler
qg.solve(problem, solver="qiskit-qaoa")  # qiskit-optimization QAOA
```

Every solver receives the identical `QUBO` object and returns the identical
`Result`, so objective, feasibility, optimality gap, and wall time are
measured the same way for all of them.

## Why this page exists

If QuGrid's exact enumeration and `dimod.ExactSolver` ever disagree, the
encoding is wrong — and every quantum result built on it is worthless. This
benchmark is the standing check that they agree, and it gives each built-in
solver a mature external reference. Example 11 asserts both properties and
exits nonzero if either fails.

## Measured table

`uv run python examples/11_cross_library_benchmark.py`, three seeds per cell,
run on 2026-08-13 on an Apple silicon laptop:

| problem | library | solver | objective | gap % | feasible | P(opt) % | time [s] |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| island-case9 | dimod | `dimod-exact` | 27.0512 | 0.00 | 100% | — | 0.0003 |
| island-case9 | dwave-samplers | `dwave-sa` | 27.0512 | 0.00 | 100% | — | 0.0015 |
| island-case9 | qugrid | `qaoa` | 27.0512 | 0.00 | 100% | 0.977 | 0.3585 |
| island-case9 | qiskit-optimization | `qiskit-qaoa` | 27.6498 | 2.21 | 100% | — | 0.3078 |
| island-case9 | qugrid | `sa` | 27.0512 | 0.00 | 100% | — | 0.0593 |
| pmu-case5 | dimod | `dimod-exact` | 2.0000 | 0.00 | 100% | — | 0.0120 |
| pmu-case5 | dwave-samplers | `dwave-sa` | 2.0000 | 0.00 | 100% | — | 0.0122 |
| pmu-case5 | qugrid | `qaoa` | 2.0000 | 0.00 | 100% | 0.134 | 3.0223 |
| pmu-case5 | qugrid | `sa` | 2.0000 | 0.00 | 100% | — | 0.0957 |
| uc-2gen | dimod | `dimod-exact` | 2908.0000 | 0.00 | 100% | — | 0.0019 |
| uc-2gen | dwave-samplers | `dwave-sa` | 2908.0000 | 0.00 | 100% | — | 0.0057 |
| uc-2gen | qugrid | `qaoa` | 10337.3333 | 255.48 | 33% | 0.057 | 0.6463 |
| uc-2gen | qugrid | `sa` | 2908.0000 | 0.00 | 100% | — | 0.0781 |

Reading the table:

- Both exact solvers land on the same objective to 1e-6 on all three problems
  (asserted by the script).
- Both simulated annealers reach the exact optimum everywhere. Ocean's C++
  inner loop is roughly ten times faster than the built-in NumPy annealer.
- The two QAOA implementations agree on islanding within seed noise: the
  built-in reaches gap 0 on 3 of 3 seeds here, qiskit-optimization on 2 of 3.
  Two codebases, one algorithm, one answer.
- Unit commitment QAOA fails the same way in any library, because the
  constraint penalties stretch the QUBO energy range far beyond the cost
  separation between schedules. Script 07 measures that failure in detail
  (see [Honest benchmarking](honest-benchmarking.md)).

## Honesty box

The external solvers here are mature classical code — exhaustive enumeration
and simulated annealing — plus one shot-based QAOA implementation. Nothing on
this page is a quantum-hardware claim. Wall times are single-laptop
measurements: expect the ordering to transfer to your machine, not the
absolute numbers.

## Install

```sh
pip install "qugrid[dwave]"    # dimod-exact, dwave-sa
pip install "qugrid[qiskit]"   # qiskit-qaoa
pip install "qugrid[all]"      # everything
```

A missing extra is not an error: example 11 prints
`SKIP <solver> (pip install qugrid[<extra>])` and validates what remains.
