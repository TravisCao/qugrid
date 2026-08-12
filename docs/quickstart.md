# 10-minute quickstart

## Install

```bash
pip install qugrid            # core: NumPy, SciPy, matplotlib, pandas — no quantum SDK
```

Or with [uv](https://docs.astral.sh/uv/): `uv add qugrid`. Optional stacks come as extras: `qugrid[qiskit]`, `qugrid[dwave]`, `qugrid[pennylane]`, `qugrid[pandapower]`, or `qugrid[all]`.

Check the install:

```bash
qugrid doctor
```

## Solve your first problem — three lines

Controlled islanding of the IEEE 9-bus system: after a disturbance, split the grid into two self-sufficient islands, cutting as little capacity as possible while keeping generation and load balanced inside each island.

```python
import qugrid as qg

result = qg.solve(qg.problems.Islanding(qg.cases.case9()), solver="qaoa", seed=0)
print(result.summary())
```

`solve()` built a QUBO (one binary variable per bus), ran the QAOA algorithm on the built-in statevector simulator, decoded the best measured bitstring back into an islanding plan, and compared it against the exact optimum, which it computed by enumeration because the problem is small enough.

## Read the result

```python
result.decoded          # engineering answer: island sets, cut lines, MW imbalance
result.feasible         # original constraints satisfied?
result.gap()            # relative distance to the classical optimum (0.0 = optimal)
result.success_probability()  # chance one measurement returns the best state
result.resources        # qubits, runtime, optimizer iterations
```

Two habits this library will keep reinforcing: judge the *decoded engineering answer*, not the bitstring; and never report a quantum number without the classical reference that ships in the same `Result`.

## See it

```python
import matplotlib.pyplot as plt

qg.viz.use_style()
d = result.decoded
qg.viz.plot_network(qg.cases.case9(), islands=d["islands"], cut_edges=d["cut_lines"])
plt.savefig("islands.png", dpi=150, bbox_inches="tight")
```

## Try the other solvers on the same problem

```python
prob = qg.problems.Islanding(qg.cases.case9())
for s in ("exact", "sa", "qaoa"):
    r = qg.solve(prob, solver=s, seed=0)
    print(f"{s:6s} objective={r.objective:10.3f} gap={r.gap():.2e} feasible={r.feasible}")
```

Same problem object, three solvers, one comparison — that is the library's core loop.

## Or run everything at once

```bash
qugrid demo --figure islands.png
```

## Next

- New to quantum computing? → [the 10-minute primer](primers/quantum-for-power.md), then Tutorial 01.
- New to power systems? → [the reverse primer](primers/power-for-quantum.md), then Tutorial 02.
- Ready to experiment? → [the example zoo](zoo.md): ten complete studies with classical references.
