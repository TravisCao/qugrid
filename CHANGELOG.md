# Changelog

All notable changes to QuGrid are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-08-13

First public release.

### Added

- `Network` data model with MATPOWER column semantics; MATPOWER `.m` parser;
  pandapower converter; seven bundled IEEE test cases (5–118 bus) and a 3-bus
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
