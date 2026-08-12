"""CLI smoke tests: `qugrid demo` and `qugrid doctor`.

Both commands finish in well under a second (demo solves the 9-bus islanding
problem exactly, with SA, and with QAOA; doctor just checks imports), so they
are run as real subprocesses -- exactly what a user types -- rather than
in-process ``main()`` calls.
"""

from __future__ import annotations

import subprocess
import sys

import qugrid as qg

TIMEOUT_S = 60


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "qugrid.cli", *args],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )


def test_demo_seed0_exits_zero_and_matches_exact_optimum():
    out = _run("demo", "--seed", "0")
    assert out.returncode == 0, out.stderr
    assert "Loaded Network" in out.stdout
    assert "Formulated Islanding" in out.stdout
    assert "solver=exact" in out.stdout
    assert "solver=sa" in out.stdout
    assert "solver=qaoa" in out.stdout
    # seed=0 is deterministic: QAOA reaches the exact islanding optimum
    assert "QAOA matches the exact optimum." in out.stdout


def test_demo_save_writes_figure(tmp_path):
    fig_path = tmp_path / "islanding.png"
    out = _run("demo", "--seed", "0", "--save", str(fig_path))
    assert out.returncode == 0, out.stderr
    assert fig_path.exists() and fig_path.stat().st_size > 0
    assert f"figure saved to {fig_path}" in out.stdout


def test_doctor_exits_zero_and_reports_environment():
    out = _run("doctor")
    assert out.returncode == 0, out.stderr
    assert f"qugrid {qg.__version__}" in out.stdout
    assert "python" in out.stdout and "numpy" in out.stdout
    # core (non-optional) dependencies are always importable in this env
    for name in ("scipy", "matplotlib", "pandas"):
        assert f"[ok]      {name:<12}" in out.stdout
    # optional adapters are reported either way, ok or missing, never crash
    for name in ("qiskit", "dimod", "pennylane", "pandapower"):
        assert name in out.stdout
    assert "[ok]      bundled cases" in out.stdout


def test_no_command_prints_help_and_exits_zero():
    out = _run()
    assert out.returncode == 0
    assert "usage" in out.stdout.lower()
    assert "demo" in out.stdout and "doctor" in out.stdout
