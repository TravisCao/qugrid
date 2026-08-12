# Quantum computing for power engineers

You work with power flow, unit commitment, and contingency analysis. This page gives you the quantum vocabulary you need to read the rest of QuGrid — in ten minutes, with power system objects in every example. Every statement here can be checked by running the code next to it.

## What a quantum computer computes

A quantum computer with $n$ qubits stores a **state**: a vector of $2^n$ complex numbers called amplitudes, one per length-$n$ bitstring. Computation applies unitary matrices to this vector. Measurement returns one bitstring, drawn with probability equal to the squared magnitude of its amplitude.

That is the entire machine model. QuGrid's simulator is exactly this — a NumPy array of $2^n$ complex numbers and functions that apply structured matrices to it (`qugrid/solvers/statevector.py`, about 200 lines; read it once and the mystery is gone).

The power system connection: a bitstring is a decision. For controlled islanding of the 9-bus system, bit $i$ says which island bus $i$ joins, so the state assigns an amplitude to every possible split of the grid at once. The algorithm's job is to concentrate amplitude on good splits before measuring.

```python
import qugrid as qg

net = qg.cases.case9()
problem = qg.problems.Islanding(net)   # 9 buses -> 9 binary variables
result = qg.solve(problem, solver="qaoa", seed=0, p=2)
for bits, prob, energy in result.top_states[:3]:
    print(bits, f"p={prob:.3f}", f"objective={energy:.1f}")
```

## The five terms you actually need

**Qubit.** One binary decision variable, held in superposition until measured. Nine buses to assign means nine qubits.

**Superposition.** The state vector holds amplitudes for all $2^n$ assignments simultaneously. This is bookkeeping, not magic: one matrix-vector product advances every assignment at once, but reading out destroys all of it except one sample.

**Measurement.** Sampling a bitstring with probability $|\text{amplitude}|^2$. Quantum algorithms are therefore judged by *success probability* — the chance a single run returns the answer you want. QuGrid reports it on every statevector result (`result.success_probability()`).

**Gate.** A unitary matrix applied to the state, usually touching one or two qubits. Circuit depth (gates applied in sequence) is the resource that noisy hardware cannot afford; that is why every result carries `result.resources`.

**Ansatz.** A parameterized circuit family. Variational algorithms (QAOA, VQE, VQLS) tune the parameters with a classical optimizer so the output state concentrates on good answers — structurally the same loop as tuning a controller: simulate, evaluate, adjust.

## The three problem shapes quantum algorithms accept

Everything quantum optimization consumes today is one of these; QuGrid's layer-2 encodings exist to translate power problems into them.

| Shape | Power system examples | Quantum algorithms |
|---|---|---|
| QUBO / Ising: minimize $x^\top Q x$, $x \in \{0,1\}^n$ | unit commitment, islanding, PMU placement, reconfiguration | quantum annealing, QAOA, VQE |
| Linear system: solve $Ax=b$ | DC power flow, the Newton step of AC power flow | HHL, VQLS |
| Kernel / distribution learning | security screening, scenario generation | quantum kernels, quantum Boltzmann machines |

Constraints do not survive the translation to QUBO; they become quadratic penalty terms. The penalty weight trade-off (too small: infeasible optimum; too large: cost differences drowned) is the single most common source of wrong conclusions in this literature. QuGrid's formulations set documented defaults and `decode()` always reports constraint violation in engineering units, so you can see what the encoding did to your problem.

## What today's hardware is

As of 2026: roughly 100–1000 noisy qubits (gate error rates around $10^{-3}$), or a few thousand analog annealing qubits with restricted coupling graphs. **No published experiment shows a quantum device beating a tuned classical solver on any power system problem.** Claims to the contrary compare against weak baselines — the failure mode QuGrid's benchmark runner is built to prevent (see [Honest benchmarking](../honest-benchmarking.md)).

The research that *is* publishable now: how encodings scale (qubit counts, penalty conditioning), where algorithmic error comes from (QuGrid's HHL reports phase-discretization leakage and postselection cost separately), what problem structure survives hardware constraints, and which grid tasks would benefit *if* fault-tolerant machines arrive. That is the work this library serves.

## Where to go next

1. [Tutorial 01 — hello, QuGrid](../tutorials.md): solve your first grid QUBO in 15 minutes.
2. [Tutorial 03 — quantum optimization 101](../tutorials.md): what QAOA actually iterates, with the landscape plot.
3. [Problem-to-algorithm cheatsheet](cheatsheet.md): one table from your problem to a runnable script.
