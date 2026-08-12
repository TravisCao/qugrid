"""Classical reference solvers.

Every quantum result in QuGrid can be checked against a classical reference,
because a quantum method you cannot validate is a demo, not research. This
subpackage holds the classical side: textbook power flow and dispatch
algorithms, implemented plainly in NumPy.
"""

from qugrid.classical.dispatch import economic_dispatch, solve_uc_enumerate
from qugrid.classical.power_flow import newton_raphson, solve_dc

__all__ = ["solve_dc", "newton_raphson", "economic_dispatch", "solve_uc_enumerate"]
