# Honest benchmarking

Quantum computing papers in power systems have a credibility problem: too many claims dissolve on contact with a tuned classical baseline. QuGrid is built so that the honest experiment is the path of least resistance. Six rules, each enforced by an API you get for free.

## 1. The power problem is the protagonist

A study is about unit commitment, not about QAOA. State results in engineering units — dispatch in MW, cost in \$, PMU counts, angle errors in degrees.

*In QuGrid:* `result.decoded` is the engineering answer; `result.summary()` prints it next to the solver metadata. The bitstring is available but never the headline.

## 2. Classical baselines run in the same script

Every quantum result ships with the answer a classical method gives on the identical instance. For QUBOs that is exact enumeration (n ≤ 24) and seeded simulated annealing; for linear systems, LU; for kernels, RBF with the same downstream classifier.

*In QuGrid:* `solve(prob, solver="exact"/"sa"/"numpy")` consume the same problem object, and `result.gap()` is always measured against the classical reference — never against another quantum run.

## 3. Separate the encoding error from the solver error

A QUBO built by discretizing generator output has already paid an approximation cost before any solver runs. Reporting only the final cost mixes two unrelated errors: what the encoding gave away, and what the solver failed to find.

*In QuGrid:* `UnitCommitment.continuous_reference()` and `EconomicDispatchQUBO.continuous_reference()` give the pre-discretization optimum; `reference()` gives the exact optimum of the QUBO itself. The difference between them is the encoding's price; the solver's gap is measured against `reference()`. `examples/05` is a complete study of exactly this separation.

## 4. Report resources, not just accuracy

“HHL solved it” means little without the postselection cost. A hardware experiment keeps only the runs where the ancilla measures 1 — at success probability 0.003, the average answer costs about 330 circuit executions.

*In QuGrid:* every result carries `resources` (qubit counts, runtime, iterations); HHL additionally reports `success_probability` and `clock_leakage`, VQLS its optimizer trace, QAOA its expectation history and per-restart behavior.

## 5. Fix seeds; sweep them

A single stochastic run is an anecdote. Fix the seed so a reader can reproduce your number, then sweep seeds so the number means something.

*In QuGrid:* every stochastic solver takes `seed=`; `bench.sweep(problems, solvers, seeds=range(10))` returns a tidy DataFrame with one row per (problem, solver, seed), and `bench.save_run()` writes results, summary, LaTeX table, and a config snapshot (library version, platform, timestamp) in one call.

## 6. Do not claim what you did not measure

The phrase “quantum advantage” does not belong in a paper whose largest instance has 22 binary variables. What research-scale instances support: statements about encodings, error sources, resource scaling, and algorithm behavior.

*In QuGrid:* nothing enforces honesty in your prose — but the README, every tutorial, and every zoo script model the phrasing, and the FAQ answers the “so is quantum useful for grids?” question the way we believe the evidence supports.

---

These rules compress the advice of the power systems reviewing community into defaults. If you publish with QuGrid, the methods section is mostly written: cite the library version from `config.json`, the seeds from your sweep, and the baselines from the same table your quantum numbers came from.

The [cross-library benchmark](cross-library.md) applies the same rules across dimod, Ocean, and Qiskit: same QUBO objects, same `Result` metrics, one table.
