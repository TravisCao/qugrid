# Changelog

All notable changes to QuGrid are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [0.2.0] — 2026-08-13

Constraint handling and solver baselines, from the power system quantum
optimization literature. Every method ships with a measured effect on a bundled
problem — see the new [Methods from the literature](https://traviscao.github.io/qugrid/methods/)
docs page, produced and asserted by examples 12 and 13.

### Added

- Slack-free inequality constraints:
  `QUBOBuilder.add_inequality(..., method="unbalanced", lam=(l1, l2))`
  (Montanez-Barrera 2024). On PJM 5-bus PMU placement: 5 qubits instead of 15,
  depth-2 QAOA success probability 0.54 instead of 0.001.
- `qugrid.methods.AugmentedLagrangianLoop`: outer multiplier loop for equality
  constraints (Hong 2025, Feng 2023). Reaches the feasible unit-commitment
  optimum in 3 outer iterations with a 21x flatter QUBO than the fixed penalty.
- `QUBO.dynamic_range()` diagnostic; `qugrid.solve` warns when a sampling
  solver receives a QUBO with dynamic range above 1e3.
- Greedy repair post-processing: `qg.solve(..., repair="greedy")` repairs the
  answer and every measured state, and reports `P(optimum | repaired)`;
  problems expose a `constraint_residual` hook (REGRID-QAOA, Jiang 2026).
- New classical baselines: tabu search (`solver="tabu"`), parallel tempering
  (`solver="pt"`), and random sampling plus repair (`solver="random+repair"`)
  — the no-quantum control every hybrid pipeline must beat (Gaidai 2026).
  All reach gap 0 on the three bundled problems in under 0.04 s.
- QAOA variants: warm start (`warm_start=` any [0,1]^n vector or
  `"relaxation"`, Egger 2021), XY one-hot mixer (`mixer=XYMixer(groups)`,
  Wang 2020), fixed transferred angles (`angles=(gammas, betas)`, Jing 2023);
  every optimized run reports its best angles in `resources`.
- Mixer protocol and `apply_unitary` on the statevector core.
- Solve-through external solvers returning the standard `Result`:
  `solver="dimod-exact"`, `"dwave-sa"`, `"qiskit-qaoa"`.
- Examples 11 (cross-library benchmark), 12 (constraint handling study),
  13 (QAOA variants study); example 07 grown to seven solvers.

### Changed

- README evidence table and solver descriptions carry the new measured
  numbers; docs navigation gains "Methods from the literature" and the
  cross-library benchmark page.

## [0.1.0] — 2026-08-13

First public release.

### Added

- `Network` data model with MATPOWER column semantics; MATPOWER `.m` parser;
  pandapower converter; seven bundled standard test cases (PJM 5-bus to IEEE 118-bus) and a 3-bus
  teaching microgrid.
- Classical references: DC power flow, polar Newton–Raphson AC power flow,
  KKT economic dispatch, exact unit-commitment enumeration.
- Problem formulations: `UnitCommitment`, `EconomicDispatchQUBO`, `Islanding`,
  `PMUPlacement`, `dc_power_flow`, `newton_with_linear_solver`,
  N-1 `screening_dataset`, toy wind scenarios; `QUBOBuilder` for custom
  formulations.
- Exact encodings: `QUBO` ⇄ `Ising` (strictly upper-triangular couplings),
  `LinearSystemProblem` with power-of-two padding and Hermitian dilation.
- Solvers on a pure-NumPy statevector core (≤ 22 qubits): QAOA, VQE, exact
  enumeration, seeded simulated annealing, random baseline; HHL (with honest
  success-probability and clock-leakage reporting) and VQLS for linear
  systems; fidelity quantum kernels with bandwidth-aware defaults; a
  visible-unit quantum Boltzmann machine.
- `solve()` front door, `Result` with `decoded` engineering answers,
  `gap()` against classical references, and resource reporting.
- Adapters: Qiskit (`SparsePauliOp`, parameterized QAOA circuit), D-Wave
  dimod BQM, PennyLane Hamiltonian.
- Benchmark runner (`bench.sweep` → tidy DataFrame → LaTeX) with
  reproducibility metadata.
- Plotting module with a colorblind-validated palette; network, convergence,
  schedule, QUBO, and distribution plots.
- CLI: `qugrid demo`, `qugrid doctor`.
- Ten self-validating example scripts; five executed tutorial notebooks;
  bilingual quickstarts; primers for both audiences; mkdocs-material site.
