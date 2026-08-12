# Choose your learning path

Three entry points, by what you already know. Times are honest estimates for a focused session.

## Path A — power system researcher, new to quantum computing

You know MATPOWER or pandapower; the word “ansatz” means nothing yet.

| Step | What | Time | You leave with |
|---|---|---|---|
| 1 | [10-minute quickstart](quickstart.md) | 10 min | the library runs on your machine; you solved an islanding QUBO |
| 2 | [Quantum computing for power engineers](primers/quantum-for-power.md) | 10 min | the five terms, the machine model, the honest hardware picture |
| 3 | Tutorial 01 — hello, QuGrid | 15 min | you can read a `Result` and explain `gap()` and `success_probability()` |
| 4 | Tutorial 03 — quantum optimization 101 | 45 min | you know what QAOA iterates and can interpret its landscape plot |
| 5 | `examples/03` (islanding) and `examples/02` (unit commitment) | 20 min | two complete studies you can modify for your own cases |
| 6 | Tutorial 05 + `examples/08` if you work on ML | 45 min | quantum kernels on an N-1 screening task, with the bandwidth lesson |

After this path you can formulate your own problem with `QUBOBuilder` (Tutorial 02 shows every step).

## Path B — quantum researcher, new to power systems

You can derive QAOA; you have never seen a bus admittance matrix.

| Step | What | Time | You leave with |
|---|---|---|---|
| 1 | [Power systems for quantum researchers](primers/power-for-quantum.md) | 10 min | the vocabulary, the credibility rules, the scale reality table |
| 2 | [10-minute quickstart](quickstart.md) | 10 min | working install |
| 3 | Tutorial 02 — from MATPOWER to QUBO | 30 min | how grid constraints become penalties, and what that costs |
| 4 | `examples/01` (HHL on DC power flow) and Tutorial 04 | 40 min | the linear-algebra side: conditioning, clock bits, postselection |
| 5 | [Honest benchmarking](honest-benchmarking.md) | 10 min | the reporting standard power venues expect |
| 6 | `examples/07` (benchmark runner) | 15 min | a seed-swept comparison table ready for a paper |

After this path you know which grid problems are worth your algorithm ideas — and which comparisons will get a paper rejected.

## Path C — you know both; you are here to do research

| Step | What | You leave with |
|---|---|---|
| 1 | [Problem-to-algorithm cheatsheet](primers/cheatsheet.md) | the API map |
| 2 | `examples/07` + `bench` module | your experiment harness: tidy DataFrame → LaTeX |
| 3 | `examples/05` (discretization study) | the encoding-vs-solver error separation, ready to extend |
| 4 | Adapters (`to_qiskit_operator`, `to_bqm`, `to_pennylane`) | the same encodings on vendor stacks and hardware |
| 5 | [Contributing](contributing.md) | where a new formulation or solver slots in (one file each) |

The library's design contract: formulations own the physics, encodings own the algebra, solvers own the search. If your research adds one of the three, you write one file and the rest of the machinery (references, gaps, benchmarks, plots) applies to it automatically.
