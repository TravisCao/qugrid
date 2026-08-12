# Power systems for quantum researchers

You know QAOA, HHL, and kernel methods. This page gives you the power system vocabulary you need to work on grid problems without misreading them — and, more importantly, the constraints that decide whether a formulation is credible to power engineers.

## The object of study

A transmission network is a graph. **Buses** (nodes) carry generation and load in megawatts (MW). **Branches** (edges) are lines and transformers with an impedance; power flows over them according to physics, not routing — you cannot send power along a chosen path. **Generators** attach to buses and have cost curves, typically quadratic: $c_2 p^2 + c_1 p + c_0$ in \$/h with $p$ in MW.

The community's standard data format is the MATPOWER case: three matrices (`bus`, `branch`, `gen`) with fixed column meanings. QuGrid's `Network` keeps those exact semantics, which is why a researcher's own case file loads without translation:

```python
import qugrid as qg

net = qg.cases.case9()          # bundled WSCC 9-bus test system
net.bus.shape, net.branch.shape  # MATPOWER columns, unchanged
```

The IEEE test cases (9, 14, 30, 39, 57, 118 buses, all bundled) are the community's shared benchmark instances — the equivalent of MaxCut on 3-regular graphs, except each one is a real physical system with published provenance.

## The four problems QuGrid encodes, and why they matter

**Power flow.** Given injections, find bus voltages such that Kirchhoff's laws hold. The DC approximation is a linear system $B'\theta = P$ (sparse, symmetric, condition numbers typically $10^2$–$10^4$); the full AC problem is solved by Newton's method, a sequence of such linear systems. This is the entry point for quantum linear solvers — and the reason `condition_number()` is on every `LinearSystemProblem`: HHL's cost scales with it.

**Unit commitment (UC).** Decide which generators are on in each hour (binary) and their output (continuous), minimizing cost subject to meeting demand. The mixed binary-continuous structure is why it becomes a QUBO only after discretizing power output — an approximation with a measurable cost that QuGrid reports separately (`continuous_reference()`).

**Controlled islanding.** After a disturbance, split the grid into self-sufficient islands to stop cascading failures. Graph partitioning with a power-balance constraint — the cleanest grid-native QUBO, and QuGrid's default demo.

**Security screening.** Operators check thousands of operating points against the **N-1 criterion**: the system must survive any single component outage. Each check is cheap; the volume motivates learned classifiers — the natural grid application for quantum kernels.

## What makes a grid formulation credible

These are the standards power system reviewers apply; violating them is why quantum papers get rejected at power venues.

1. **Engineering units end to end.** A result is a dispatch in MW and a cost in \$/h, not an Ising energy. QuGrid's `decode()` exists for this.
2. **Constraint violations quantified.** “The ground state was feasible” is not enough; report the balance error in MW. Penalty encodings make this non-optional.
3. **The baseline is the field's actual tool.** DC power flow is solved by sparse LU in microseconds; UC by mixed-integer programming at national scale nightly. A quantum comparison against random guessing or unoptimized enumeration convinces nobody.
4. **Test systems are standard.** Results on IEEE cases are checkable; results on a hand-made 4-bus graph are not.
5. **N-1 thinking.** Any operational claim must survive single-outage scrutiny; it is the field's default notion of robustness.

## Scale reality

| Problem | Realistic research instance | Variables / qubits | Production instance |
|---|---|---|---|
| DC power flow | 9–118 bus IEEE cases | system size 8–117, needs $\lceil \log_2 n \rceil$ + clock qubits in HHL | 10k–80k buses |
| Unit commitment | 2–5 units, 2–4 periods as QUBO | 10–22 binary vars | 1k units, 36–48 periods |
| Islanding | 9–39 bus cases | one var per bus | 10k buses |
| Screening | 3–8 load features | one qubit per feature | hundreds of features |

The gap between columns two and four is the honest headline: research-scale quantum experiments inform algorithm design; they do not run grids. QuGrid keeps you productive in column two and precise about the distance to column four.

## Where to go next

1. [Tutorial 02 — from a MATPOWER case to a QUBO](../tutorials.md): the full encoding pipeline, including what penalty weights do.
2. [Example zoo](../zoo.md): ten runnable studies, each with a classical reference.
3. [Honest benchmarking](../honest-benchmarking.md): the reporting rules this library enforces.
