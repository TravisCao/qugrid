# Methods from the literature

Every method on this page comes from a published paper with a power system
application, ships as a first-class QuGrid API, and carries a measured
effect on a bundled problem. The numbers are produced and asserted by
[example 12](https://github.com/TravisCao/qugrid/blob/main/examples/12_constraint_handling_study.py)
and [example 13](https://github.com/TravisCao/qugrid/blob/main/examples/13_qaoa_variants_study.py)
— run them to reproduce everything below (measured 2026-08-13).

## Unbalanced penalization for inequalities

**What it is.** The textbook route for an inequality constraint on an
annealer adds binary slack variables until the constraint becomes an
equality, then squares it. The slack register costs qubits — PMU placement
on the PJM 5-bus system needs 10 slack bits on top of 5 placement bits.
Unbalanced penalization replaces the slack construction with the asymmetric
penalty `-l1*g + l2*g^2` on the constraint function `g >= 0`: zero extra
qubits, at the price of an approximation whose penalty minimum does not
exactly coincide with the constraint boundary.

**API.**

```python
builder.add_inequality(terms, constant, weight=w)                          # slack (default)
builder.add_inequality(terms, constant, method="unbalanced", lam=(1, 2))   # slack-free
```

**Papers.** A. Lucas, "Ising formulations of many NP problems", Frontiers
in Physics 2, 5 (2014), arXiv:1302.5843 (the slack construction).
J. A. Montanez-Barrera, D. Willsch, A. Maldonado-Romo, K. Michielsen,
"Unbalanced penalization: a new approach to encode inequality constraints
of combinatorial problems for quantum optimization algorithms", Quantum
Science and Technology 9, 025022 (2024), arXiv:2211.13914.

**Measured effect (example 12).** PMU placement, PJM 5-bus, depth-2 QAOA,
seed 0:

| encoding   | qubits | dynamic range | P(optimum) | gap |
|------------|-------:|--------------:|-----------:|----:|
| slack      | 15     | 4.0           | 0.0014     | 0%  |
| unbalanced | 5      | 3.5           | 0.5364     | 0%  |

Both encodings agree with exhaustive enumeration (2 PMUs). The 383x
success-probability gain is the removed slack register. Caveat: the
`lam` pair needs per-problem-family tuning — the knapsack values from the
paper (0.96, 0.0371) do *not* transfer to this coverage problem (they
return 5 PMUs); the QuGrid default `(1, 1)` works here. Always check
`is_feasible` on the returned answer.

## Augmented Lagrangian outer loop for equalities

**What it is.** A fixed quadratic penalty forces a bad choice: too small
returns infeasible answers, large enough for exactness stretches the QUBO
coefficient range until sampling solvers cannot resolve the cost
differences underneath. The augmented Lagrangian keeps the quadratic weight
small and moves constraint pressure into linear multiplier terms,
re-solving the QUBO between multiplier updates
(`u += 2*lam*g(x*)`, `lam *= alpha`).

**API.**

```python
from qugrid.methods import AugmentedLagrangianLoop
loop = AugmentedLagrangianLoop(objective_qubo, constraints, solver="sa")
result = loop.run(seed=0)      # standard Result; loop.history has every iterate
```

**Papers.** J. Hong, Y. Xu, F. Teng, "Quantum annealing-aided aggregated
unit commitment", arXiv:2502.15917. W. Feng, Y. Zhang, M. A. Bragin,
Y. Zhou, "Scalability and performance of quantum computing for unit
commitment", IEEE Transactions on Power Systems 38(3), 2023.

**Measured effect (example 12).** Unit commitment, 2 units x 2 periods,
demand [60, 130] MW:

| method               | dynamic range | gap  | feasible |
|----------------------|--------------:|-----:|----------|
| fixed penalty (QAOA p=2) | 3.9e3     | 383% | no       |
| augmented Lagrangian (SA inner) | 180 | 0%  | yes      |

The loop reaches the feasible optimum (cost $2908) in 3 outer iterations;
its objective QUBO stays 21x flatter than the fixed-penalty encoding. The
`qugrid.solve` front door warns (`RuntimeWarning`) whenever a sampling
solver receives a QUBO with `dynamic_range() > 1e3`, and
`QUBO.dynamic_range()` is the diagnostic behind that warning.

## Greedy repair and the random+repair baseline

**What it is.** Sampling solvers put probability mass on infeasible
bitstrings whenever penalties compete with costs. Greedy repair
post-processes each sample: flip the bit (or bit pair, when single flips
stall) that most reduces the problem's `constraint_residual`, until
feasible. The honest control is uniform random sampling pushed through the
same repair — a hybrid pipeline that cannot beat `random+repair` has no
quantum content.

**API.**

```python
qg.solve(prob, solver="qaoa", repair="greedy")   # repairs answer + distribution
qg.solve(prob, solver="random+repair")           # the baseline to beat
result.extras["p_optimum_raw"], result.extras["p_optimum_repaired"]
```

**Papers.** Z. Jiang et al., "REGRID-QAOA" (grid islanding on IBM
hardware, 9-57 buses, repair as the core contribution), arXiv:2606.15083.
I. Gaidai, A. Mukherjee (negative result: hybrid pipelines at 5-13 qubits
failed to beat random sampling plus classical evaluation),
arXiv:2607.15543.

**Measured effect (examples 07 and 12).** Repair reaches feasibility from
every uniform random state on all three bundled formulations (tested over
seeded batches). On unit commitment it turns depth-2 QAOA's
`P(optimum) = 0.000` into `P(optimum | repaired) = 0.011`, and turns plain
random sampling (gap 77%) into an exact solver (gap 0) at 256 samples. On
islanding and PMU placement `random+repair` equals plain random sampling —
almost every uniform sample is already feasible there, and repair restores
feasibility, it does not optimize.

## Tabu search and parallel tempering

**What it is.** Stronger classical opponents than exhaustive enumeration
and plain simulated annealing. Tabu search is deterministic best-move
local search with a recency memory; parallel tempering runs a ladder of
Metropolis replicas with exchange moves, the strongest widely reported
classical heuristic on sparse QUBOs. Any quantum claim built on QuGrid
should survive both.

**API.**

```python
qg.solve(prob, solver="tabu", seed=0)
qg.solve(prob, solver="pt", seed=0)
```

**Papers.** The comparison practice follows P. Kaseb et al., "Quantum
annealing versus classical solvers for grid partitioning",
arXiv:2505.15978.

**Measured effect (example 07).** Both reach gap 0 on all three bundled
problems, every seed, in under 0.04 s — the honest bar that QAOA (255%
mean gap on unit commitment) does not clear.

## Warm-start QAOA

**What it is.** Vanilla QAOA starts from the uniform superposition and
must find good regions from scratch at every depth. Warm-start QAOA starts
from a product state biased toward a classical solution `c in [0,1]^n`
(qubit `i` measures 1 with probability `c_i`) and replaces the mixer with
one that keeps that state as its ground state. An `epsilon` clip keeps
every bit movable.

**API.**

```python
qg.solve(prob, solver="qaoa", warm_start=classical_x)     # any [0,1]^n vector
qg.solve(prob, solver="qaoa", warm_start="relaxation")    # built-in box relaxation
```

**Papers.** D. J. Egger, J. Marecek, S. Woerner, "Warm-starting quantum
optimization", Quantum 5, 479 (2021). A. Salgado, A. Sequeira, J. Santos
(unit commitment at p=1 within 5.1% of reference), arXiv:2412.11312.

**Measured effect (example 13).** Islanding, WSCC 9-bus, depth p=1:
`P(optimum)` rises from 0.009 (vanilla) to 0.128 warm-started at the
simulated-annealing answer — one order of magnitude from the same circuit
depth. With `epsilon=0` the start state freezes and `P(optimum) = 1`
trivially, which is why the clip defaults to 0.25.

## XY mixer for one-hot structure

**What it is.** When variables form one-hot groups (a unit picks exactly
one output level), the ring-XY mixer conserves Hamming weight inside each
group: initialize each group in its W state and every reachable state
satisfies the one-hot constraint with probability 1 — no penalty, no
slack, no infeasible samples. The honest cost is qubit count: one-hot
spends `levels x units` qubits where binary spends
`ceil(log2 levels) x units`; the built-in 22-qubit statevector ceiling
holds about 7 units at 3 levels.

**API.**

```python
from qugrid.solvers import XYMixer
qg.solve(prob, solver="qaoa", mixer=XYMixer(groups))   # groups: list of qubit lists
```

**Papers.** Z. Wang, S. Hadfield, Z. Jiang, E. G. Rieffel, "XY-mixers:
analytical and numerical results for the quantum alternating operator
ansatz", Physical Review A 101, 012320 (2020). N. Mohseni et al. (biased
variant, single-period unit commitment on IBM hardware),
arXiv:2603.00260.

**Measured effect (example 13).** 2-unit, 3-level one-hot dispatch, p=2:
the XY mixer holds 100.0% of the output distribution on one-hot-feasible
states and hits the optimum exactly (gap 0); the penalty encoding of the
same problem puts 22% of its mass on feasible states. Probability mass
outside the feasible subspace is 0 to numerical precision (asserted at
1e-9 in the test suite).

## Fixed transferred angles

**What it is.** The classical angle optimization is the expensive part of
QAOA. Angle transfer reuses angles optimized on one instance for another
instance of the same problem family, skipping the optimizer entirely.

**API.**

```python
res = qg.solve(small_instance, solver="qaoa", p=1)
qg.solve(big_instance, solver="qaoa", p=1,
         angles=(res.resources["gammas"], res.resources["betas"]))
```

Every optimized run reports its best angles in
`resources["gammas"]/["betas"]`, so transfer experiments can harvest them.

**Papers.** H. Jing, Y. Wang, Y. Li, "Data-driven quantum approximate
optimization algorithm for power systems", Communications Engineering 2,
12 (2023) — transfer keyed by normalized graph density on IEEE 24-bus
maximum power sections.

**Measured effect (example 13).** Depth-1 angles optimized on 9-bus
islanding, transferred to 14-bus islanding: expectation 211.9 against
202.9 for the full optimization (within 5%), at 1 circuit evaluation
against 929. Arbitrary angles miss by ~90%. Caveat: the published
transfer evidence is MaxCut-like; on penalty-stretched formulations it is
untested — measure before trusting.

## What QuGrid deliberately does not ship

Recursive QAOA, multi-angle QAOA, CVaR aggregation, reverse annealing,
Grover mixers, and simulated bifurcation have no power or energy
application paper in the survey behind this page; they stay out until one
exists. ADMM and Benders-style decomposition are deferred to a later
release — they deserve their own design round, not a corner of this one.
