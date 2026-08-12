"""The :class:`Network` container: a typed, convenient view of a MATPOWER case.

Design goals:

* **Zero translation cost for power system researchers.** A ``Network`` is a
  thin wrapper around the MATPOWER arrays (``bus``, ``gen``, ``branch``,
  ``gencost``). Anything that works in MATPOWER/PYPOWER/pandapower can get in
  and out losslessly.
* **Everything downstream asks the network, not the arrays.** Admittance and
  DC matrices, graph structure, load/generation vectors, and index bookkeeping
  live here, so problem formulations stay short and readable.

Units follow MATPOWER: MW/MVAr in the data arrays, per-unit internally where
noted, angles in degrees in the arrays and radians in computed quantities.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qugrid import idx


@dataclass
class Network:
    """A power network in MATPOWER case format with computed conveniences."""

    baseMVA: float
    bus: np.ndarray
    gen: np.ndarray
    branch: np.ndarray
    gencost: np.ndarray | None = None
    name: str = "network"
    _ext2int: dict[int, int] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.bus = np.atleast_2d(np.asarray(self.bus, dtype=float)).copy()
        self.gen = np.atleast_2d(np.asarray(self.gen, dtype=float)).copy()
        self.branch = np.atleast_2d(np.asarray(self.branch, dtype=float)).copy()
        if self.gencost is not None and np.size(self.gencost):
            self.gencost = np.atleast_2d(np.asarray(self.gencost, dtype=float)).copy()
        else:
            self.gencost = None
        self._ext2int = {int(b): i for i, b in enumerate(self.bus[:, idx.BUS_I])}
        if len(self._ext2int) != self.n_bus:
            raise ValueError("duplicate bus numbers in case data")

    # ------------------------------------------------------------------ sizes
    @property
    def n_bus(self) -> int:
        return self.bus.shape[0]

    @property
    def n_gen(self) -> int:
        return self.gen.shape[0]

    @property
    def n_branch(self) -> int:
        return self.branch.shape[0]

    # ------------------------------------------------------------- index maps
    def bus_index(self, external: int | np.ndarray) -> np.ndarray | int:
        """Map external bus number(s) to internal 0-based position(s)."""
        if np.isscalar(external):
            return self._ext2int[int(external)]
        return np.array([self._ext2int[int(b)] for b in np.asarray(external).ravel()])

    @property
    def f_bus(self) -> np.ndarray:
        """Internal index of each branch's from-bus."""
        return self.bus_index(self.branch[:, idx.F_BUS])

    @property
    def t_bus(self) -> np.ndarray:
        """Internal index of each branch's to-bus."""
        return self.bus_index(self.branch[:, idx.T_BUS])

    @property
    def gen_bus(self) -> np.ndarray:
        """Internal bus index of each generator."""
        return self.bus_index(self.gen[:, idx.GEN_BUS])

    @property
    def branch_on(self) -> np.ndarray:
        return self.branch[:, idx.BR_STATUS] > 0

    @property
    def gen_on(self) -> np.ndarray:
        return self.gen[:, idx.GEN_STATUS] > 0

    @property
    def ref(self) -> int:
        """Internal index of the reference (slack) bus."""
        refs = np.flatnonzero(self.bus[:, idx.BUS_TYPE] == idx.REF)
        if len(refs) == 0:
            raise ValueError("case has no reference bus")
        return int(refs[0])

    @property
    def pv(self) -> np.ndarray:
        return np.flatnonzero(self.bus[:, idx.BUS_TYPE] == idx.PV)

    @property
    def pq(self) -> np.ndarray:
        return np.flatnonzero(self.bus[:, idx.BUS_TYPE] == idx.PQ)

    # ------------------------------------------------------------ injections
    @property
    def load_p(self) -> np.ndarray:
        """Active load per bus [MW]."""
        return self.bus[:, idx.PD].copy()

    @property
    def load_q(self) -> np.ndarray:
        """Reactive load per bus [MVAr]."""
        return self.bus[:, idx.QD].copy()

    def gen_p_per_bus(self) -> np.ndarray:
        """Scheduled in-service generation per bus [MW]."""
        p = np.zeros(self.n_bus)
        on = self.gen_on
        np.add.at(p, self.gen_bus[on], self.gen[on, idx.PG])
        return p

    def sbus(self) -> np.ndarray:
        """Complex net power injection per bus [pu]."""
        sload = (self.bus[:, idx.PD] + 1j * self.bus[:, idx.QD]) / self.baseMVA
        sgen = np.zeros(self.n_bus, dtype=complex)
        on = self.gen_on
        np.add.at(
            sgen,
            self.gen_bus[on],
            (self.gen[on, idx.PG] + 1j * self.gen[on, idx.QG]) / self.baseMVA,
        )
        return sgen - sload

    # ---------------------------------------------------------------- Y / Bdc
    def ybus(self) -> np.ndarray:
        """Dense complex bus admittance matrix [pu].

        Follows MATPOWER's ``makeYbus`` (branch pi model with off-nominal tap
        ratio and phase shift). Dense is a deliberate choice: QuGrid targets
        research-scale cases (up to a few hundred buses), where dense algebra
        is simpler and fast enough.
        """
        nb, nl = self.n_bus, self.n_branch
        stat = self.branch_on.astype(float)
        ys = stat / (self.branch[:, idx.BR_R] + 1j * self.branch[:, idx.BR_X])
        bc = stat * self.branch[:, idx.BR_B]
        tap_mag = np.where(self.branch[:, idx.TAP] == 0.0, 1.0, self.branch[:, idx.TAP])
        tap = tap_mag * np.exp(1j * np.deg2rad(self.branch[:, idx.SHIFT]))
        ytt = ys + 1j * bc / 2
        yff = ytt / (tap * np.conj(tap))
        yft = -ys / np.conj(tap)
        ytf = -ys / tap
        ysh = (self.bus[:, idx.GS] + 1j * self.bus[:, idx.BS]) / self.baseMVA
        f, t = self.f_bus, self.t_bus
        y = np.zeros((nb, nb), dtype=complex)
        np.add.at(y, (f, f), yff)
        np.add.at(y, (t, t), ytt)
        np.add.at(y, (f, t), yft)
        np.add.at(y, (t, f), ytf)
        y[np.arange(nb), np.arange(nb)] += ysh
        del nl
        return y

    def bdc(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """DC power flow matrices, following MATPOWER's ``makeBdc``.

        Returns ``(Bbus, Bf, Pbusinj, Pfinj)`` such that, in per unit,
        ``P_bus = Bbus @ theta + Pbusinj`` and ``P_branch = Bf @ theta + Pfinj``.
        """
        nb = self.n_bus
        stat = self.branch_on.astype(float)
        b = stat / self.branch[:, idx.BR_X]
        tap_mag = np.where(self.branch[:, idx.TAP] == 0.0, 1.0, self.branch[:, idx.TAP])
        b = b / tap_mag
        f, t = self.f_bus, self.t_bus
        nl = self.n_branch
        bf = np.zeros((nl, nb))
        bf[np.arange(nl), f] += b
        bf[np.arange(nl), t] -= b
        bbus = np.zeros((nb, nb))
        np.add.at(bbus, (f, f), b)
        np.add.at(bbus, (t, t), b)
        np.add.at(bbus, (f, t), -b)
        np.add.at(bbus, (t, f), -b)
        pfinj = b * (-np.deg2rad(self.branch[:, idx.SHIFT]))
        pbusinj = np.zeros(nb)
        np.add.at(pbusinj, f, pfinj)
        np.add.at(pbusinj, t, -pfinj)
        return bbus, bf, pbusinj, pfinj

    # ------------------------------------------------------------------ graph
    def edges(self, in_service_only: bool = True) -> list[tuple[int, int, float]]:
        """Branch list as ``(from, to, |b|)`` with internal indices.

        The weight is the branch susceptance magnitude ``1/x`` — the standard
        measure of electrical coupling used in partitioning studies.
        """
        out = []
        for line in range(self.n_branch):
            if in_service_only and not self.branch_on[line]:
                continue
            w = abs(1.0 / self.branch[line, idx.BR_X])
            out.append((int(self.f_bus[line]), int(self.t_bus[line]), float(w)))
        return out

    def adjacency(self) -> np.ndarray:
        """Symmetric 0/1 adjacency matrix of the in-service network."""
        a = np.zeros((self.n_bus, self.n_bus))
        for f, t, _ in self.edges():
            a[f, t] = a[t, f] = 1.0
        return a

    def is_connected(self) -> bool:
        a = self.adjacency()
        seen = np.zeros(self.n_bus, dtype=bool)
        stack = [0]
        seen[0] = True
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(a[u]):
                if not seen[v]:
                    seen[v] = True
                    stack.append(int(v))
        return bool(seen.all())

    # ------------------------------------------------------------- transforms
    def copy(self) -> Network:
        return Network(
            baseMVA=self.baseMVA,
            bus=self.bus,
            gen=self.gen,
            branch=self.branch,
            gencost=self.gencost,
            name=self.name,
        )

    def scale_loads(self, factor: float | np.ndarray) -> Network:
        """Return a copy with active and reactive loads multiplied by ``factor``.

        ``factor`` is a scalar or a per-bus array of length ``n_bus``.
        """
        out = self.copy()
        out.bus[:, idx.PD] *= factor
        out.bus[:, idx.QD] *= factor
        return out

    def drop_branch(self, line: int) -> Network:
        """Return a copy with one branch switched out (N-1 outage)."""
        out = self.copy()
        out.branch[line, idx.BR_STATUS] = 0
        return out

    # ------------------------------------------------------------ constructors
    @classmethod
    def from_ppc(cls, ppc: dict, name: str = "network") -> Network:
        """Build from a MATPOWER/PYPOWER case dict (``baseMVA, bus, gen, branch``)."""
        return cls(
            baseMVA=float(ppc["baseMVA"]),
            bus=ppc["bus"],
            gen=ppc["gen"],
            branch=ppc["branch"],
            gencost=ppc.get("gencost"),
            name=name,
        )

    @classmethod
    def from_matpower(cls, path: str, name: str | None = None) -> Network:
        """Read a MATPOWER ``.m`` case file."""
        from pathlib import Path

        from qugrid.io.matpower import read_matpower

        ppc = read_matpower(path)
        return cls.from_ppc(ppc, name=name or Path(path).stem)

    @classmethod
    def from_pandapower(cls, net, name: str | None = None) -> Network:
        """Convert a pandapower network (requires ``pandapower`` installed)."""
        try:
            from pandapower.converter import to_ppc
        except ImportError as err:  # pragma: no cover - exercised without extra
            raise ImportError(
                "pandapower is required for Network.from_pandapower; "
                "install with `pip install qugrid[pandapower]`"
            ) from err
        ppc = to_ppc(net, init="flat")
        return cls.from_ppc(ppc, name=name or "pandapower-net")

    def to_ppc(self) -> dict:
        """Export as a PYPOWER-compatible case dict."""
        ppc = {
            "version": "2",
            "baseMVA": self.baseMVA,
            "bus": self.bus.copy(),
            "gen": self.gen.copy(),
            "branch": self.branch.copy(),
        }
        if self.gencost is not None:
            ppc["gencost"] = self.gencost.copy()
        return ppc

    # ------------------------------------------------------------------ report
    def __repr__(self) -> str:
        return (
            f"Network({self.name!r}: {self.n_bus} buses, {self.n_branch} branches, "
            f"{self.n_gen} generators, load {self.load_p.sum():.1f} MW)"
        )
