"""The ``qugrid`` command line: instant demo, environment doctor.

``qugrid demo`` is the 30-second first contact: split the WSCC 9-bus system
into two self-sufficient islands with QAOA, check the answer against exact
enumeration and simulated annealing, and (optionally) save the figure.
"""

from __future__ import annotations

import argparse
import sys


def _cmd_demo(args: argparse.Namespace) -> int:
    import qugrid as qg

    net = qg.cases.case9()
    print(f"Loaded {net!r}")
    problem = qg.problems.Islanding(net)
    print(f"Formulated {problem!r}\n")
    results = {}
    for solver in ("exact", "sa", "qaoa"):
        results[solver] = qg.solve(problem, solver=solver, seed=args.seed)
        print(results[solver].summary())
        print()
    if args.save:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        res = results["qaoa"]
        ax = qg.viz.plot_network(
            net,
            islands=res.decoded["islands"],
            cut_edges=res.decoded["cut_lines"],
            title="QAOA islanding of the WSCC 9-bus system",
        )
        ax.figure.savefig(args.save)
        plt.close("all")
        print(f"figure saved to {args.save}")
    qaoa, exact = results["qaoa"], results["exact"]
    same = abs((qaoa.objective or 0) - (exact.objective or 1)) < 1e-6
    match = "matches" if same else "differs from"
    print(f"QAOA {match} the exact optimum. Next: docs/tutorials, or `import qugrid`.")
    return 0


def _cmd_doctor(_args: argparse.Namespace) -> int:
    import numpy

    import qugrid

    py = sys.version.split()[0]
    print(f"qugrid {qugrid.__version__} | python {py} | numpy {numpy.__version__}")
    optional = ("scipy", "matplotlib", "pandas", "qiskit", "dimod", "pennylane", "pandapower")
    for name in optional:
        try:
            mod = __import__(name)
            print(f"  [ok]      {name:<12} {getattr(mod, '__version__', '?')}")
        except ImportError:
            extra = {
                "qiskit": "qugrid[qiskit]",
                "dimod": "qugrid[dwave]",
                "pennylane": "qugrid[pennylane]",
                "pandapower": "qugrid[pandapower]",
            }.get(name, name)
            print(f"  [missing] {name:<12} install: pip install {extra}")
    n = qugrid.cases.case9().n_bus
    print(f"  [ok]      bundled cases ({n}-bus smoke check passed)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qugrid",
        description="Quantum computing for power system research.",
    )
    sub = parser.add_subparsers(dest="command")
    demo = sub.add_parser("demo", help="run the 30-second QAOA islanding demo")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--save", type=str, default="", help="save the network figure to this path")
    sub.add_parser("doctor", help="check the installation and optional extras")
    args = parser.parse_args(argv)
    if args.command == "demo":
        return _cmd_demo(args)
    if args.command == "doctor":
        return _cmd_doctor(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
