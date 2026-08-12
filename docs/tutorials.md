# Tutorials

Five executed notebooks in `notebooks/` take you from zero to running your own studies. Each states its audience and time budget; markdown outweighs code in all of them. Open them on GitHub, or run locally:

```bash
git clone https://github.com/TravisCao/qugrid && cd qugrid
uv sync --extra dev
uv run --with jupyterlab jupyter lab notebooks/
```

| # | Notebook | Time | For whom | You leave with |
|---|---|---|---|---|
| 01 | `01_hello_qugrid` | 15 min | everyone; zero quantum knowledge assumed | your first solved grid QUBO and the `Result` vocabulary: `decoded`, `gap()`, `feasible`, `success_probability()` |
| 02 | `02_from_matpower_to_qubo` | 30 min | anyone with their own case data | the full encoding pipeline: MATPOWER columns → `Network` → hand-built QUBO with `QUBOBuilder` → what penalty weights do (both failure directions, demonstrated numerically) |
| 03 | `03_quantum_optimization_101` | 45 min | power engineers who want to *understand* QAOA/VQE | the mechanics: amplitudes as island assignments, the (γ, β) landscape plot, distribution sharpening with depth, and why SA still wins at this scale |
| 04 | `04_quantum_linear_solvers` | 30 min | anyone touching power flow | DC power flow as $Ax=b$; HHL error anatomy (clock bits, fidelity vs relative error vs success probability); the hybrid Newton loop |
| 05 | `05_qml_for_screening` | 30 min | ML-inclined researchers | N-1 screening as a classification task; quantum kernel vs RBF with the bandwidth experiment; QBM scenario generation |

Suggested orders per background are in [learning paths](learning-paths.md).
