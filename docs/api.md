# API reference

The public surface is small by design: one data model, one problem layer, one `solve()` front door.

## Front door

::: qugrid.solvers.solve

## Data model

::: qugrid.network.Network

## Problem formulations

::: qugrid.problems.unit_commitment.UnitCommitment

::: qugrid.problems.economic_dispatch.EconomicDispatchQUBO

::: qugrid.problems.islanding.Islanding

::: qugrid.problems.pmu.PMUPlacement

::: qugrid.problems.power_flow.dc_power_flow

::: qugrid.problems.power_flow.newton_with_linear_solver

::: qugrid.problems.screening.screening_dataset

## Encodings

::: qugrid.problems.base.QUBO

::: qugrid.problems.base.Ising

::: qugrid.problems.base.LinearSystemProblem

::: qugrid.problems.builder.QUBOBuilder

## Results and benchmarking

::: qugrid.solvers.base.Result

::: qugrid.bench

## Classical references

::: qugrid.classical.power_flow

::: qugrid.classical.dispatch

## Adapters

::: qugrid.adapters
