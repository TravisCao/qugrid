# Contributing

QuGrid grows in the direction its users' research goes. The design contract makes most contributions a single file.

## What to contribute, by value

1. **A problem formulation from your research.** One file in `src/qugrid/problems/`, subclassing `CombinatorialProblem` (implement `qubo`, `decode`, `is_feasible`) or producing a `LinearSystemProblem`. Every solver, the benchmark runner, and the plotting utilities apply to it automatically. Ship it with one test comparing the QUBO ground state against an independent classical formulation — the pattern in `tests/test_formulations.py`.
2. **A zoo script.** A single self-validating file in `examples/` following the contract in [the zoo page](zoo.md): docstring (problem / why quantum / expected output), classical baseline in the same run, final assertion, one or two figures.
3. **Error analysis.** Deeper honesty metrics for an existing solver (e.g. shot-noise models, hardware-noise studies via the adapters).
4. **Tutorial translations.** Especially Chinese; keep the English version the source of truth.

## Ground rules

- **Honest benchmarking is non-negotiable.** Contributions that compare a quantum method only against weak baselines, or state advantage claims the run does not measure, will be asked to revise. The [six rules](honest-benchmarking.md) are the review checklist.
- **No new heavy dependencies in the core.** NumPy/SciPy/matplotlib/pandas only; anything vendor-specific goes behind an extra in `adapters/`.
- **Conventions are frozen:** QUBO minimizes $x^\top Q x$; spins $s = 1 - 2x$; `Ising.j` strictly upper triangular; bit $i$ of a basis index is variable $i$. `tests/test_encodings.py` enforces them at $10^{-9}$.

## Workflow

```bash
git clone https://github.com/TravisCao/qugrid && cd qugrid
uv sync --extra dev
uv run pytest tests/ -q     # green before you start
uv run ruff check .         # and after every change
```

Pull requests need: passing CI, a test for any behavior change, and a one-paragraph motivation written for a power system researcher (what question does this answer?).

## Reporting problems

Open a GitHub issue with the library version (`qugrid doctor`), a minimal script, and — for numerical disagreements — the classical reference you compared against.
