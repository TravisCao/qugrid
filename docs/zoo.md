# Example zoo

Thirteen single-file studies in `examples/`, in the spirit of CleanRL: each is self-contained, seeded, runnable in minutes, ends by validating its own claim against a classical reference, and saves its figures to `examples/figures/`.

```bash
uv run python examples/03_islanding_case9.py
```

| # | Script | Power problem | Quantum method | Classical reference |
|---|---|---|---|---|
| 01 | `01_dc_power_flow_hhl` | DC power flow (3-bus microgrid) | HHL | LU / `solve_dc` |
| 02 | `02_unit_commitment_qaoa` | 2-unit, 2-period unit commitment | QAOA | exact UC enumeration |
| 03 | `03_islanding_case9` | controlled islanding, WSCC 9-bus | QAOA | exact + simulated annealing |
| 04 | `04_pmu_placement` | PMU observability, 9/14-bus | SA on QUBO | exact dominating set |
| 05 | `05_economic_dispatch_discretization` | dispatch discretization study | exact QUBO | KKT economic dispatch |
| 06 | `06_qaoa_depth_study` | islanding | QAOA at p = 1…4 | exact optimum |
| 07 | `07_solver_benchmark` | UC + islanding + PMU | QAOA vs SA vs random, seed-swept | exact; emits LaTeX table |
| 08 | `08_quantum_kernel_screening` | N-1 security screening | fidelity kernel (bandwidth study) | RBF kernel |
| 09 | `09_qbm_wind_scenarios` | wind scenario generation | quantum Boltzmann machine | empirical moments |
| 10 | `10_hybrid_newton_vqls` | AC power flow | VQLS inner solves | Newton–Raphson |
| 11 | `11_cross_library_benchmark` | UC + islanding + PMU | QAOA in two implementations | dimod exact + Ocean SA, same `Result`; emits LaTeX table |
| 12 | `12_constraint_handling_study` | PMU coverage + UC balance constraints | slack vs unbalanced penalties, augmented Lagrangian, greedy repair | exhaustive enumeration over both encodings |
| 13 | `13_qaoa_variants_study` | islanding (9- and 14-bus) + one-hot dispatch | warm-start QAOA, XY mixer, transferred angles | exact optimum + simulated annealing warm starter |

`examples/README.md` carries the same table with measured runtimes and the exact figures each script produces.

## The contract every script honors

1. The docstring states the power problem, why quantum methods are studied for it (one honest sentence), and the expected output.
2. The classical baseline runs in the same file, on the same instance, and the script *asserts* its own headline claim — if the claim stops holding, the script exits nonzero and CI notices.
3. Figures use the library's plotting style; no result exists only as a picture.

Copy any script, swap in your case file, and you have the skeleton of a study.
