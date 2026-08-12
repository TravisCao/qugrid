"""QuGrid: quantum computing for power system research.

From a MATPOWER case to a quantum algorithm in three lines:

>>> import qugrid as qg
>>> problem = qg.problems.Islanding(qg.cases.case9())
>>> result = qg.solve(problem, solver="qaoa", seed=0)
>>> print(result.summary())

Layers (each usable on its own):

* :mod:`qugrid.cases` / :class:`qugrid.Network` — IEEE test systems and
  MATPOWER/pandapower interchange.
* :mod:`qugrid.problems` — power problems encoded as QUBOs / linear systems.
* :mod:`qugrid.solvers` — built-in exact simulators and classical baselines.
* :mod:`qugrid.adapters` — export to Qiskit, D-Wave Ocean, PennyLane.
* :mod:`qugrid.bench` / :mod:`qugrid.viz` — paper-ready experiments and plots.
"""

from qugrid import adapters, bench, cases, classical, problems, solvers, viz
from qugrid._version import __version__
from qugrid.network import Network
from qugrid.solvers import Result, solve

__all__ = [
    "__version__",
    "Network",
    "solve",
    "Result",
    "cases",
    "problems",
    "solvers",
    "classical",
    "adapters",
    "bench",
    "viz",
]
