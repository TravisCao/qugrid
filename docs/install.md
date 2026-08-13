# Install

## Requirements

Python 3.10 or newer. The core depends only on NumPy, SciPy, matplotlib, and pandas — no quantum SDK, no compiler, no account.

## Standard install

```bash
pip install qugrid
```

With [uv](https://docs.astral.sh/uv/) (recommended for research projects):

```bash
uv add qugrid
```

## Optional extras

| Extra | Installs | Unlocks |
|---|---|---|
| `qugrid[qiskit]` | qiskit | `adapters.to_qiskit_operator`, `to_qiskit_qaoa` — IBM stack and hardware |
| `qugrid[dwave]` | dimod | `adapters.to_bqm` — D-Wave Ocean and annealing hardware |
| `qugrid[pennylane]` | pennylane | `adapters.to_pennylane` — differentiable-circuit workflows |
| `qugrid[pandapower]` | pandapower | `Network.from_pandapower` |
| `qugrid[all]` | all of the above | everything |

Calling an adapter without its extra raises an `ImportError` that names the exact command to run.

## Verify

```bash
qugrid doctor    # versions, optional-dependency status
qugrid demo      # 30-second end-to-end run on the WSCC 9-bus system
```

## Development install

```bash
git clone https://github.com/TravisCao/qugrid && cd qugrid
uv sync --extra dev        # test + lint toolchain
uv run pytest tests/ -q    # should be green
uv run ruff check .        # should be clean
```
