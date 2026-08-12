# Problem-to-algorithm cheatsheet

Start from your power problem in the left column; every row ends in code you can run today.

## Optimization problems

| Power problem | Mathematical shape | QuGrid formulation | Quantum solvers | Classical references (same run) | Runnable example |
|---|---|---|---|---|---|
| Unit commitment | binary commitment + discretized dispatch → QUBO | `problems.UnitCommitment(gens, demand, power_bits)` | `"qaoa"`, `"vqe"`, annealing via D-Wave adapter | `"exact"` (n ≤ 20), `"sa"`, `solve_uc_enumerate` | `examples/02` |
| Economic dispatch (study of discretization) | binary-expansion QUBO | `problems.EconomicDispatchQUBO(gens, demand, power_bits)` | `"qaoa"`, `"vqe"` | `"exact"`, `"sa"`, KKT `economic_dispatch` | `examples/05` |
| Controlled islanding | graph cut + power balance → QUBO | `problems.Islanding(net)` | `"qaoa"`, `"vqe"` | `"exact"`, `"sa"` | `examples/03` |
| PMU placement | dominating set → QUBO with slack bits | `problems.PMUPlacement(net)` | `"qaoa"`, annealing | `"exact"`, `"sa"`, enumeration | `examples/04` |

## Linear-algebra problems

| Power problem | Mathematical shape | QuGrid formulation | Quantum solvers | Classical reference | Runnable example |
|---|---|---|---|---|---|
| DC power flow | $B'\theta = P$, symmetric, sparse | `problems.dc_power_flow(net)` | `"hhl"`, `"vqls"` | `"numpy"` (LU), `classical.solve_dc` | `examples/01` |
| AC power flow (hybrid) | Newton iteration; each step solves $J \Delta = -f$ | `problems.newton_with_linear_solver(net, solve_fn)` | inner `"vqls"` / `"hhl"` | `classical.newton_raphson` | `examples/10` |

## Machine-learning problems

| Power problem | Mathematical shape | QuGrid formulation | Quantum method | Classical baseline | Runnable example |
|---|---|---|---|---|---|
| N-1 security screening | binary classification | `problems.screening_dataset(net)` | fidelity quantum kernel (`solvers.quantum_kernel`) | RBF kernel, same classifier | `examples/08` |
| Renewable scenario generation | distribution learning | `problems.toy_wind_profiles` + `binarize` | quantum Boltzmann machine (`solvers.QuantumBoltzmannMachine`) | empirical moments | `examples/09` |

## Choosing solver options

| You want | Do this |
|---|---|
| The true optimum for a small QUBO (n ≤ 20) | `qg.solve(prob, solver="exact")` — enumerates, always right |
| A strong classical baseline at any size | `solver="sa"`, raise `n_restarts` before trusting a gap |
| To study QAOA itself | `solver="qaoa", p=1..4, restarts≥3`; read `resources["expectation"]` and `success_probability()` |
| Linear-solver error anatomy | `solver="hhl", n_clock=4..10`; read `relative_error`, `success_probability`, `clock_leakage` |
| Real hardware / vendor stacks | `adapters.to_qiskit_operator`, `adapters.to_bqm`, `adapters.to_pennylane` — same encoding objects |

## Reading a `Result`

| Field | Meaning | Credibility rule |
|---|---|---|
| `decoded` | engineering answer (MW, \$, bus sets) | this is the result; the bitstring is not |
| `gap()` | relative distance to the classical reference optimum | report it; 0.0 means optimal |
| `feasible` | original constraints satisfied (pre-penalty) | never report objective without it |
| `success_probability()` | chance one measurement returns the best state | the honest cost of sampling algorithms |
| `resources` | qubits, runtime, iterations, solver internals | scale claims live here |
