"""io.matpower parsing and the Network container.

Covers every bundled case (shapes, Ybus/Bdc, injections, graph structure),
the transforms every problem formulation builds on (scale_loads, drop_branch),
the from_ppc/to_ppc round trip, and the parser's error messages on
non-MATPOWER input.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

import qugrid as qg
from qugrid import idx
from qugrid.io.matpower import parse_matpower, read_matpower
from qugrid.network import Network

BUNDLED = ("case5", "case9", "case14", "case30", "case39", "case57", "case118")


@pytest.mark.parametrize("name", BUNDLED)
def test_bundled_case_shapes_are_consistent(name):
    net = qg.cases.load_case(name)
    assert net.baseMVA > 0
    assert net.n_bus > 0 and net.n_branch > 0 and net.n_gen > 0
    assert net.bus.shape == (net.n_bus, net.bus.shape[1]) and net.bus.shape[1] >= 13
    assert net.gen.shape == (net.n_gen, net.gen.shape[1]) and net.gen.shape[1] >= 10
    assert net.branch.shape == (net.n_branch, net.branch.shape[1]) and net.branch.shape[1] >= 11


@pytest.mark.parametrize("name", BUNDLED)
def test_ybus_symmetric_when_no_phase_shifters(name):
    net = qg.cases.load_case(name)
    shift = net.branch[:, idx.SHIFT]
    if np.any(shift != 0):
        pytest.skip(f"{name} has phase-shifting transformers; Ybus need not be symmetric")
    y = net.ybus()
    assert y.shape == (net.n_bus, net.n_bus)
    assert np.allclose(y, y.T, atol=1e-10)


@pytest.mark.parametrize("name", BUNDLED)
def test_bdc_shapes(name):
    net = qg.cases.load_case(name)
    bbus, bf, pbusinj, pfinj = net.bdc()
    assert bbus.shape == (net.n_bus, net.n_bus)
    assert bf.shape == (net.n_branch, net.n_bus)
    assert pbusinj.shape == (net.n_bus,)
    assert pfinj.shape == (net.n_branch,)


@pytest.mark.parametrize("name", BUNDLED)
def test_sbus_sums_to_net_injection(name):
    net = qg.cases.load_case(name)
    s = net.sbus()
    assert s.shape == (net.n_bus,)
    on = net.gen_on
    expected_p = (net.gen[on, idx.PG].sum() - net.load_p.sum()) / net.baseMVA
    expected_q = (net.gen[on, idx.QG].sum() - net.load_q.sum()) / net.baseMVA
    assert s.sum().real == pytest.approx(expected_p, abs=1e-9)
    assert s.sum().imag == pytest.approx(expected_q, abs=1e-9)


@pytest.mark.parametrize("name", BUNDLED)
def test_edges_count_matches_in_service_branches(name):
    net = qg.cases.load_case(name)
    edges = net.edges()
    assert len(edges) == int(net.branch_on.sum())
    # every bundled case ships with all branches in service
    assert len(edges) == net.n_branch


@pytest.mark.parametrize("name", BUNDLED)
def test_adjacency_is_symmetric(name):
    net = qg.cases.load_case(name)
    a = net.adjacency()
    assert a.shape == (net.n_bus, net.n_bus)
    assert np.array_equal(a, a.T)


def test_scale_loads_scales_active_and_reactive_load_linearly(case9):
    factor = 1.7
    scaled = case9.scale_loads(factor)
    assert scaled.load_p == pytest.approx(case9.load_p * factor)
    assert scaled.load_q == pytest.approx(case9.load_q * factor)

    per_bus = np.linspace(0.5, 1.5, case9.n_bus)
    scaled_per_bus = case9.scale_loads(per_bus)
    assert scaled_per_bus.load_p == pytest.approx(case9.load_p * per_bus)

    # scale_loads returns a copy; the original network is untouched
    assert case9.load_p == pytest.approx(case9.load_p)


def test_drop_branch_marks_one_line_out_without_shrinking_the_array(case9):
    line = 0
    before_in_service = int(case9.branch_on.sum())
    out = case9.drop_branch(line)

    # n_branch counts array rows (needed for stable N-1 bookkeeping across a
    # whole outage study), not in-service lines, so it is unchanged; the
    # *effective* topology loses exactly the dropped edge.
    assert out.n_branch == case9.n_branch
    assert not out.branch_on[line]
    assert int(out.branch_on.sum()) == before_in_service - 1
    assert len(out.edges()) == len(case9.edges()) - 1

    # every other branch row is untouched
    other = np.arange(case9.n_branch) != line
    assert np.array_equal(out.branch[other], case9.branch[other])

    # drop_branch works on a copy: the original network is not mutated
    assert int(case9.branch_on.sum()) == before_in_service


def test_from_ppc_to_ppc_roundtrip_case9(case9):
    ppc = case9.to_ppc()
    net2 = Network.from_ppc(ppc, name="case9")
    assert net2.baseMVA == case9.baseMVA
    assert np.allclose(net2.bus, case9.bus)
    assert np.allclose(net2.gen, case9.gen)
    assert np.allclose(net2.branch, case9.branch)
    assert case9.gencost is not None and net2.gencost is not None
    assert np.allclose(net2.gencost, case9.gencost)


def test_parse_matpower_error_on_missing_required_fields(tmp_path):
    bad = tmp_path / "not_a_case.m"
    bad.write_text(
        "%% not a real MATPOWER case: no bus/gen/branch arrays at all\nmpc.version = '2';\n"
    )
    with pytest.raises(ValueError, match=r"mpc\.baseMVA"):
        read_matpower(bad)
    with pytest.raises(ValueError, match=r"mpc\.baseMVA"):
        Network.from_matpower(str(bad))


def test_parse_matpower_error_on_matlab_expressions(tmp_path):
    bad = tmp_path / "matlab_code.m"
    bad.write_text(
        "mpc.baseMVA = 100;\n"
        "mpc.bus = [\n"
        "    1 x+1 0 0 0 0 1 1.0 0 230 1 1.1 0.9;\n"
        "];\n"
    )
    with pytest.raises(ValueError, match="non-numeric"):
        parse_matpower(bad.read_text())


def test_toy3_sanity():
    net = qg.cases.toy3()
    assert net.n_bus == 3
    assert net.load_p.sum() == pytest.approx(150.0)


@pytest.mark.skipif(
    importlib.util.find_spec("pandapower") is not None,
    reason="pandapower installed; happy path not covered here",
)
def test_network_from_pandapower_missing_package_hint():
    with pytest.raises(ImportError, match=r"qugrid\[pandapower\]"):
        Network.from_pandapower(object())
