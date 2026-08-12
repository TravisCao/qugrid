# FAQ

## Is quantum computing useful for power systems today?

For operating grids: no. Sparse LU factorization solves DC power flow in microseconds; commercial mixed-integer programming solves national-scale unit commitment nightly. No published experiment shows a quantum device beating these tools, and QuGrid will not tell you otherwise.

For research: yes, and that is the library's purpose. Which grid problems have structure quantum algorithms could exploit, what encodings cost, where algorithmic error comes from, what resources realistic instances demand — these questions are answerable now, publishable now, and they determine whether the field is ready if fault-tolerant hardware arrives.

## Why a power-specific library instead of Qiskit Optimization or D-Wave Ocean?

Those stacks consume abstract QUBOs and return bitstrings. The research work in power applications happens on both sides of that interface: turning a MATPOWER case into a *credible* QUBO (discretization, penalty weights, constraint bookkeeping), and turning a bitstring back into a dispatch with quantified constraint violation, compared against the field's actual baselines. QuGrid owns those two sides and hands the middle to any stack you like via adapters.

## Why is the core simulator limited to 22 qubits?

A statevector of $2^n$ complex doubles at $n=22$ is 64 MB and a dense-ish operation touches all of it; beyond that, laptops stop being interactive. The limit is a friendly error, not a hidden truncation. Larger combinatorial instances run through simulated annealing (any size) or the D-Wave adapter; larger statevector experiments through the Qiskit and PennyLane adapters on their optimized simulators.

Research honesty note: if your scientific claim only appears above 22 qubits, it needs hardware evidence anyway.

## Why does the QUBO ground state not match the cheapest feasible dispatch?

By design, and this is documented in every formulation: the default penalty weights make constraint satisfaction dominate, so the ground state minimizes the *balance error* first and cost second (a lexicographic preference). When demand is exactly representable on the discretization grid, the two orderings agree. `decode()` always reports `balance_error_mw` so nothing is hidden; Tutorial 02 demonstrates both failure directions (weights too small and too large).

## My QAOA gap is worse than simulated annealing. Is something broken?

No — that is the expected result on most instances, and pretending otherwise is how the field got its credibility problem. QAOA at small depth is a research object, not a competitive optimizer. Study its expectation-vs-depth behavior (`examples/06`), its success probability, its parameter landscape (Tutorial 03) — those are the publishable quantities.

## Can I use my own grid data?

Yes, three ways: `Network.from_matpower("case.m")` for MATPOWER files, `Network.from_pandapower(net)` with the pandapower extra, or `Network.from_ppc(dict)` for in-memory MATPOWER-style dicts. Bus/branch/gen arrays keep MATPOWER column semantics (constants in `qugrid.idx`).

## Do I need a quantum computer or an account with a vendor?

No. The core runs entirely on NumPy. When you do have access, the adapters export the identical problem objects to Qiskit, D-Wave Ocean, and PennyLane, so the study you prototyped locally runs on hardware without re-encoding.

## How do I cite QuGrid?

See `CITATION.cff` in the repository root; GitHub renders a “Cite this repository” button from it. Also cite the algorithm papers referenced in each solver's docstring — HHL, QAOA, VQLS, and the quantum-power-flow literature are not this library's contribution.

## What should I contribute?

The highest-value additions, in order: a new problem formulation from your own research (one file in `problems/`, following the `CombinatorialProblem` contract); a real dataset or case study for the zoo; error analysis of an existing solver; translations of the tutorials. See [Contributing](contributing.md).
