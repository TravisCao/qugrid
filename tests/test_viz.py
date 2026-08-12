"""Smoke tests for qugrid.viz.

Every public plot function is exercised on case9/toy3 (small, fast), checked
to run without error, and checked for the artifact counts that are cheap to
assert (number of axes, lines, patches). No pixel-level assertions.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import qugrid as qg
from qugrid import viz


@pytest.fixture(autouse=True)
def _close_all_figures():
    yield
    plt.close("all")


def test_use_style_runs_and_is_idempotent():
    viz.use_style()
    viz.use_style()
    assert plt.rcParams["axes.grid"] is True


def test_plot_network_plain(case9):
    ax = viz.plot_network(case9)
    assert len(ax.figure.axes) == 1
    assert len(ax.lines) == case9.n_branch  # one line per branch


def test_plot_network_with_islands(case9):
    islands = np.arange(case9.n_bus) % 2
    ax = viz.plot_network(case9, islands=islands, title="islands")
    assert ax.get_title() == "islands"
    assert len(ax.figure.axes) == 1  # categorical colors: no colorbar


def test_plot_network_with_cut_edges(case9):
    cut = [(f, t) for f, t, _ in case9.edges()[:2]]
    ax = viz.plot_network(case9, cut_edges=cut)
    assert len(ax.lines) == case9.n_branch  # cut edges are still drawn (dashed)


def test_plot_network_with_values(case9):
    values = np.linspace(0.0, 1.0, case9.n_bus)
    ax = viz.plot_network(case9, node_values=values, node_label="loading")
    assert len(ax.figure.axes) == 2  # main axes + colorbar


def test_plot_network_toy3_highlight_and_loading(toy3):
    ax = viz.plot_network(toy3, highlight_buses=[0], edge_loading=np.array([0.2, 0.5, 1.1]))
    assert len(ax.figure.axes) == 1
    assert len(ax.lines) == toy3.n_branch


def test_plot_convergence_sa_result(case9):
    res = qg.solve(qg.problems.Islanding(case9), solver="sa", seed=0)
    ax = viz.plot_convergence(res)
    assert len(ax.lines) >= 1
    assert ax.get_legend() is None  # single series: no legend


def test_plot_convergence_multiple_results_list(case9):
    prob = qg.problems.Islanding(case9)
    results = [
        qg.solve(prob, solver="sa", seed=0),
        qg.solve(prob, solver="sa", seed=1),
    ]
    ax = viz.plot_convergence(results)
    legend = ax.get_legend()
    assert legend is not None and len(legend.get_texts()) == 2


def test_plot_uc_schedule():
    gens = [
        qg.problems.GenParams("base", pmin=0, pmax=90, c2=0.02, c1=10.0, c0=50.0),
        qg.problems.GenParams("peaker", pmin=0, pmax=60, c2=0.04, c1=20.0, c0=30.0),
    ]
    demand = [60.0, 130.0]
    uc = qg.problems.UnitCommitment(gens, demand, power_bits=2)
    decoded = uc.reference()  # has "power" (n_gen x n_t), what the plot needs
    ax = viz.plot_uc_schedule(decoded, demand, gen_names=["base", "peaker"])
    assert len(ax.patches) == len(gens) * len(demand)  # one bar per unit-period
    assert len(ax.lines) == 1  # the demand step line


def test_plot_distribution_requires_sampled_states(case9):
    """Classical (exact) results have no output distribution to plot."""
    res = qg.solve(qg.problems.Islanding(case9), solver="exact")
    assert res.top_states == []
    with pytest.raises(ValueError, match="sampled states"):
        viz.plot_distribution(res)


def test_plot_distribution_on_qaoa_islanding(case9):
    res = qg.solve(
        qg.problems.Islanding(case9), solver="qaoa", seed=0, p=1, restarts=1, maxiter=60
    )
    ax = viz.plot_distribution(res)
    assert len(ax.patches) == min(12, len(res.top_states))  # default top=12


def test_plot_qubo(case9):
    prob = qg.problems.Islanding(case9)
    ax = viz.plot_qubo(prob.qubo, title="islanding QUBO")
    assert len(ax.images) == 1
    assert len(ax.figure.axes) == 2  # main axes + colorbar
