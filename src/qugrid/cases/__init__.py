"""Bundled benchmark cases.

The IEEE and PJM test systems every power system researcher already knows,
ready in one call — no MATLAB, no downloads:

>>> import qugrid as qg
>>> net = qg.cases.case14()
>>> net
Network('case14': 14 buses, 20 branches, 5 generators, load 259.0 MW)

The ``.m`` files are unmodified MATPOWER data files (BSD licensed, see
``NOTICE``). ``toy3()`` is a hand-made 3-bus microgrid used by the tutorials:
small enough that every quantum state fits on a slide.
"""

from __future__ import annotations

from importlib import resources

import numpy as np

from qugrid.io.matpower import parse_matpower
from qugrid.network import Network

_BUNDLED = ("case5", "case9", "case14", "case30", "case39", "case57", "case118")


def _load(name: str) -> Network:
    text = resources.files("qugrid.cases").joinpath(f"data/{name}.m").read_text()
    return Network.from_ppc(parse_matpower(text), name=name)


def case5() -> Network:
    """PJM 5-bus system (Li & Bo). The classic quantum power flow demo case."""
    return _load("case5")


def case9() -> Network:
    """WSCC 9-bus, 3-generator system (Chow)."""
    return _load("case9")


def case14() -> Network:
    """IEEE 14-bus test case."""
    return _load("case14")


def case30() -> Network:
    """IEEE 30-bus test case."""
    return _load("case30")


def case39() -> Network:
    """New England 39-bus, 10-generator system."""
    return _load("case39")


def case57() -> Network:
    """IEEE 57-bus test case."""
    return _load("case57")


def case118() -> Network:
    """IEEE 118-bus test case."""
    return _load("case118")


def toy3() -> Network:
    """A 3-bus microgrid for teaching: 2 generators, 1 load, 3 lines.

    Bus 1: slack with a cheap generator. Bus 2: PV bus with an expensive
    generator. Bus 3: PQ bus with a 150 MW load.
    """
    bus = np.array(
        [
            # I type Pd  Qd Gs Bs area Vm  Va baseKV zone Vmax Vmin
            [1, 3, 0.0, 0.0, 0, 0, 1, 1.0, 0, 230, 1, 1.1, 0.9],
            [2, 2, 0.0, 0.0, 0, 0, 1, 1.0, 0, 230, 1, 1.1, 0.9],
            [3, 1, 150.0, 40.0, 0, 0, 1, 1.0, 0, 230, 1, 1.1, 0.9],
        ]
    )
    gen = np.zeros((2, 21))
    #        bus  Pg    Qg  Qmax  Qmin  Vg   mBase status Pmax Pmin
    gen[0, :10] = [1, 90.0, 0.0, 100, -100, 1.0, 100, 1, 200, 0]
    gen[1, :10] = [2, 60.0, 0.0, 100, -100, 1.0, 100, 1, 100, 0]
    branch = np.zeros((3, 13))
    #           f  t   r      x     b   rateA rateB rateC tap shift status
    branch[0, :11] = [1, 2, 0.01, 0.06, 0.0, 250, 250, 250, 0, 0, 1]
    branch[1, :11] = [1, 3, 0.01, 0.08, 0.0, 250, 250, 250, 0, 0, 1]
    branch[2, :11] = [2, 3, 0.01, 0.07, 0.0, 250, 250, 250, 0, 0, 1]
    branch[:, 11] = -360.0
    branch[:, 12] = 360.0
    gencost = np.array(
        [
            # model startup shutdown ncost c2    c1   c0
            [2, 100.0, 0.0, 3, 0.02, 15.0, 100.0],
            [2, 200.0, 0.0, 3, 0.05, 30.0, 200.0],
        ]
    )
    return Network(baseMVA=100.0, bus=bus, gen=gen, branch=branch, gencost=gencost, name="toy3")


def load_case(name: str) -> Network:
    """Load a bundled case by name (``"case14"``) or a ``.m`` file by path."""
    if name in _BUNDLED:
        return _load(name)
    if name == "toy3":
        return toy3()
    return Network.from_matpower(name)


__all__ = [
    "case5",
    "case9",
    "case14",
    "case30",
    "case39",
    "case57",
    "case118",
    "toy3",
    "load_case",
]
