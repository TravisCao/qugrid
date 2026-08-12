# QuGrid examples

Ten single-file scripts. Each one takes a power system problem, solves it with a
quantum algorithm, compares the answer against the classical baseline in
engineering units, prints a table, writes its figures, and asserts the claim it
makes. Every script is seeded, so the numbers in its docstring are the numbers
you get.

None of these scripts claims a quantum speed advantage. All of the instances are
small enough to solve exactly, which is what makes the comparison meaningful:
the honest measurement is the product.

## How to run

```sh
uv run python examples/06_qaoa_depth_study.py
```

Any script runs the same way. Figures go to `examples/figures/`, and script 07
also writes a benchmark run to `examples/runs/benchmark/`. A script exits with
status 0 when every assertion holds and with status 1 when one fails, so the
whole set runs as a check:

```sh
for f in examples/*.py; do uv run python "$f" || echo "FAIL $f"; done
```

The full set takes about 40 seconds.

## Index

| Script | Power problem | Quantum method | Classical baseline | Runtime | Figures |
| --- | --- | --- | --- | --- | --- |
| `01_dc_power_flow_hhl.py` | DC power flow of the 3-bus microgrid `toy3` | HHL linear solver over clock register sizes | `numpy.linalg.solve` and the DC power flow reference | 1 s | `01_dc_power_flow_hhl_error.png`, `01_dc_power_flow_hhl_network.png` |
| `02_unit_commitment_qaoa.py` | Unit commitment, 2 thermal units over 2 periods with startup costs | QAOA over circuit depth | exhaustive enumeration with exact economic dispatch, and simulated annealing | 3 s | `02_unit_commitment_qaoa_schedule.png`, `02_unit_commitment_qaoa_solvers.png` |
| `03_islanding_case9.py` | Controlled islanding of the WSCC 9-bus system | depth-2 QAOA | exhaustive enumeration over 512 states, and simulated annealing | 1 s | `03_islanding_case9_network.png`, `03_islanding_case9_distribution.png` |
| `04_pmu_placement.py` | Minimum PMU placement on the 9-bus and 14-bus systems | simulated annealing on the QUBO encoding | exact minimum dominating set by enumeration | 7 s | `04_pmu_placement_networks.png` |
| `05_economic_dispatch_discretization.py` | Economic dispatch, 2 units, 100 MW demand | binary power encoding that every quantum optimizer consumes, 1 to 4 bits per unit | continuous economic dispatch by bisection on marginal cost | 1 s | `05_economic_dispatch_discretization_error.png` |
| `06_qaoa_depth_study.py` | Controlled islanding of the WSCC 9-bus system | QAOA at depths 1, 2, 3, 4 with 3 restarts | exhaustive enumeration, and uniform random sampling for the success probability | 3 s | `06_qaoa_depth_study_depth.png`, `06_qaoa_depth_study_distribution.png` |
| `07_solver_benchmark.py` | Unit commitment, islanding, and PMU placement together | depth-2 QAOA | exhaustive enumeration, simulated annealing, and uniform random sampling, 3 seeds each | 19 s | `07_solver_benchmark_gap.png` |
| `08_quantum_kernel_screening.py` | N-1 security screening of the WSCC 9-bus system | fidelity quantum kernel with the ZZ feature map, over 3 bandwidths and 2 depths | RBF kernel with the median heuristic, same classifier | 1 s | `08_quantum_kernel_screening_bandwidth.png`, `08_quantum_kernel_screening_gram.png` |
| `09_qbm_wind_scenarios.py` | Wind scenario generation over a 5-period horizon | quantum Boltzmann machine, transverse-field Ising, exact diagonalization | empirical means and neighbor correlations of the 300 training profiles | 1 s | `09_qbm_wind_scenarios_training.png`, `09_qbm_wind_scenarios_statistics.png` |
| `10_hybrid_newton_vqls.py` | AC power flow of the 3-bus microgrid `toy3` | VQLS inside every Newton-Raphson iteration | Newton-Raphson with an LU inner solve | 3 s | `10_hybrid_newton_vqls_convergence.png`, `10_hybrid_newton_vqls_network.png` |

## What each script measures

**01** compares HHL angles against the exact DC solution and reports the
post-selection success probability against clock register size.

**02** separates the two error sources that reports of quantum unit commitment
often merge: the discretization error of the binary encoding, and the solver
error against the QUBO optimum.

**03** solves the partitioning problem with the longest record on annealing
hardware, and reports the island contents in MW and the DC flow on each opened
line.

**04** counts the qubits that the slack-variable encoding of an inequality
constraint costs: the 9-bus instance needs 9 placement bits and 15 slack bits.

**05** measures the accuracy floor that the binary power encoding sets, before
any solver runs.

**06** measures what QAOA depth buys. The cost expectation falls from 128.85 at
p = 1 to 94.26 at p = 4, and the probability of measuring an optimal partition
rises from 0.90% to 1.27%, against 0.39% for uniform random sampling.

**07** runs the loop that a benchmark paper reports: problems by solvers by
seeds, into a tidy DataFrame, a summary table, a LaTeX table, and a pinned run
configuration. Simulated annealing reaches the exact optimum on all three
problems. QAOA fails on unit commitment, where the constraint penalties stretch
the QUBO energy range to 709,000 while the cost difference between the best and
the second-best schedule is 30.

**08** shows that the angle range the features are scaled into decides the
result. Test accuracy runs from 1.000 to 0.500 across the settings tried, while
training accuracy stays at or above 0.866. The RBF baseline holds 1.000
throughout.

**09** checks a generative model on the two statistics that scenario users need:
mean output per period, and the neighbor correlation that carries ramp
structure. Both agree with the data within 0.014.

**10** replaces the linear solve inside Newton-Raphson with a variational
quantum solve. Both methods converge in 4 iterations to the same operating
point. The inner solves carry a relative error of 1.5e-3, about 7 orders of
magnitude looser than the final answer, which is the defining property of an
inexact Newton method.

## Notes on two choices

Script 07 places PMU placement on the PJM 5-bus system rather than the 9-bus
system. The 9-bus formulation needs 24 binary variables, and the built-in
statevector simulator holds at most 22 qubits, so QAOA cannot run on it. Script
04 covers the 9-bus and 14-bus instances with simulated annealing, which has no
such limit.

Script 07 writes its LaTeX table with `qugrid.bench.save_run`, whose writer
builds booktabs LaTeX by hand and needs only the core dependencies. The script
keeps a local writer as a fallback, and its output line names which writer ran.
