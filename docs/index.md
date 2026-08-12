# QuGrid

![QuGrid](assets/logo.svg){ width="560" }

**Quantum computing for power system research: from a MATPOWER case to a quantum algorithm in three lines.**

```python
import qugrid as qg

result = qg.solve(qg.problems.Islanding(qg.cases.case9()), solver="qaoa", seed=0)
print(result.summary())
```

QuGrid is a research library for power system scholars who want to study quantum algorithms without leaving their own field's tools, units, and standards of evidence.

![Architecture](assets/architecture.svg){ width="880" }

## What it gives you

- **Your data, unchanged.** MATPOWER cases, pandapower networks, and seven bundled standard test systems (PJM 5-bus to IEEE 118-bus), with MATPOWER column semantics throughout.
- **Grid problems, pre-formulated.** Unit commitment, controlled islanding, PMU placement, economic dispatch, DC/AC power flow, N-1 security screening, scenario generation — each with documented encodings and decoders back to engineering units.
- **A quantum core with zero SDK dependencies.** Exact statevector simulation of QAOA, VQE, HHL, VQLS, quantum kernels, and a quantum Boltzmann machine, in pure NumPy. No SDK, no account, no compiler.
- **Classical baselines in the same run.** Exact enumeration, simulated annealing, LU, Newton–Raphson — every `Result` carries its `gap()` against them.
- **Escape hatches to real stacks.** The same problem objects export to Qiskit, D-Wave Ocean, and PennyLane.
- **Publication plumbing.** Seed-swept benchmarks to tidy DataFrames and LaTeX tables; a colorblind-safe plotting style.

## Start here

| You are… | Go to |
|---|---|
| a power researcher new to quantum | [Quickstart](quickstart.md) → [quantum primer](primers/quantum-for-power.md) → Tutorial 01 |
| a quantum researcher new to grids | [power primer](primers/power-for-quantum.md) → Tutorial 02 |
| here to run experiments | [cheatsheet](primers/cheatsheet.md) → [example zoo](zoo.md) → `bench` |
| 中文读者 | [十分钟上手](quickstart_zh.md) |

## The one paragraph of honesty

No quantum device today beats tuned classical solvers on any power system problem, and QuGrid never implies otherwise. What the library makes easy is the research that is real now: encoding costs, error anatomy, resource scaling, and fair comparisons — with the classical baseline always in the same table. See [Honest benchmarking](honest-benchmarking.md).
