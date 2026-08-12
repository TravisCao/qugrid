# QuGrid tutorial notebooks

A guided path from a MATPOWER case file to a quantum algorithm, written for power
system researchers. The reader is assumed to know MATPOWER or pandapower and to
know no quantum computing at all. Every quantum term is defined at first use in
power system vocabulary, and every performance statement is measured inside the
notebook that makes it.

## Which notebook to read

| # | Notebook | Minutes | Assumes | You will be able to |
| --- | --- | --- | --- | --- |
| 1 | [`01_hello_qugrid`](01_hello_qugrid.ipynb) | 15 | MATPOWER case format. No quantum computing. | State what a QUBO is, encode controlled islanding as one, run three solvers on it, and read every field of a `Result`. |
| 2 | [`02_from_matpower_to_qubo`](02_from_matpower_to_qubo.ipynb) | 30 | Notebook 1 | Load your own `.m` file, read case arrays by column name, build an economic dispatch QUBO by hand, and choose penalty weights that work. |
| 3 | [`03_quantum_optimization_101`](03_quantum_optimization_101.ipynb) | 45 | Notebooks 1 and 2 | Explain amplitudes, the Ising form, and what QAOA’s two angles per layer do — including the depth-1 cost landscape drawn in full. |
| 4 | [`04_quantum_linear_solvers`](04_quantum_linear_solvers.ipynb) | 30 | Notebook 1 | Derive DC power flow as `A x = b`, solve it with HHL and VQLS, take HHL’s error apart into three numbers, and run AC power flow with a quantum inner solver. |
| 5 | [`05_qml_for_screening`](05_qml_for_screening.ipynb) | 30 | Notebook 1 | Build an N-1 security classification task from any network, compare a quantum kernel against a tuned classical one, and fit a quantum Boltzmann machine to wind scenarios. |

Notebooks 4 and 5 cover algorithm families independent of notebooks 2 and 3, so
either can be read straight after notebook 1.

## Reading order by goal

* **New to quantum computing, want the shortest complete picture** — notebook 1,
  then notebook 3. About one hour.
* **Have a power problem and want to encode it** — notebooks 1 and 2. The penalty
  weight discussion in section 2.4 is the part that decides whether a formulation
  works.
* **Interested in power flow** — notebooks 1 and 4.
* **Interested in machine learning surrogates or scenario generation** —
  notebooks 1 and 5.
* **Preparing a paper** — read the closing sections of notebooks 3, 4, and 5.
  Each one states what its experiments do and do not establish, with the classical
  baseline measured beside the quantum result.

## Running them

Every notebook is committed with its outputs, so they can be read without
running anything. To run them yourself:

```
uv run --with jupyterlab jupyter lab notebooks/
```

Each notebook executes end to end in well under a minute on a laptop, needs no
quantum SDK, and seeds every random number generator, so re-running reproduces
every figure and every number in the text.

## Source format

The `.py` files are the sources, in
[jupytext](https://jupytext.readthedocs.io/) percent format: `# %% [markdown]`
opens a text cell and `# %%` opens a code cell. They are plain Python, so they
diff and review like code. The `.ipynb` files are generated from them.

To rebuild one notebook after editing its source:

```
uv run --with jupytext,nbconvert,ipykernel jupytext --to ipynb notebooks/01_hello_qugrid.py
uv run --with jupytext,nbconvert,ipykernel jupyter nbconvert --to notebook --execute --inplace notebooks/01_hello_qugrid.ipynb
```

`uv run ruff check notebooks/` lints both formats.
