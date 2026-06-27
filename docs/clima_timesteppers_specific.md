# ClimaTimeSteppers.jl — Repo-Specific Guide

<!-- This file documents things specific to ClimaTimeSteppers.jl that are NOT
     covered by the shared DeveloperGuides at docs/dev-guides/. -->

## Package overview

ClimaTimeSteppers.jl is the time integration layer for the CliMA climate model
stack. It provides high-performance ODE solvers — IMEX Additive Runge-Kutta
(ARK), low-storage explicit Runge-Kutta (LSRK), Rosenbrock, SSP, MIS, and
Wicker-Skamarock multirate methods — together with a flexible Newton/Krylov
nonlinear solver (GMRES, Jacobian-free Newton-Krylov). The library is designed
for zero-allocation, type-stable stepping on CPUs and GPUs and is used directly
by [ClimaAtmos.jl](https://github.com/CliMA/ClimaAtmos.jl),
[ClimaLand.jl](https://github.com/CliMA/ClimaLand.jl), and
[ClimaCoupler.jl](https://github.com/CliMA/ClimaCoupler.jl).

## Directory map

Mapped to the architectural layers in
[architectural_boundaries.md](dev-guides/architecture/architectural_boundaries.md).

| Layer | Directory / File | Description |
|:---|:---|:---|
| Entry point | `src/ClimaTimeSteppers.jl` | Module definition and public exports |
| Solvers | `src/solvers/imex_ark.jl` | IMEX Additive Runge-Kutta driver and stage loop |
| Solvers | `src/solvers/imex_ssprk.jl` | IMEX strong-stability-preserving Runge-Kutta |
| Solvers | `src/solvers/lsrk.jl` | Low-storage explicit Runge-Kutta (2N storage) |
| Solvers | `src/solvers/rosenbrock.jl` | Linearly implicit Rosenbrock methods |
| Solvers | `src/solvers/mis.jl` | Multirate infinitesimal step (MIS) methods |
| Solvers | `src/solvers/multirate.jl` | Generic `Multirate` wrapper for split-explicit timestepping |
| Solvers | `src/solvers/wickerskamarock.jl` | Wicker-Skamarock multirate RK methods (`WSRK2`, `WSRK3`) |
| Solvers | `src/solvers/explicit_tableaus.jl` | Explicit RK tableau definitions |
| Solvers | `src/solvers/imex_tableaus.jl` | IMEX ARK tableau definitions (ARS, IMKG, ARK families) |
| Nonlinear solvers | `src/nl_solvers/newtons_method.jl` | Newton's method with GMRES/JFNK, line search, convergence |
| Integrator | `src/integrators.jl` | `TimeStepperIntegrator` state, `init`, `step!`, `solve!` dispatch |
| ODE functions | `src/functions.jl` | `ClimaODEFunction` (T_exp!, T_lim!, T_imp!, lim!, dss!, cache!) |
| Problems | `src/problems.jl` | `ODEProblem` definition |
| Utilities | `src/utilities/fused_increment.jl` | Fused in-place update (`@fused_increment`) |
| Utilities | `src/utilities/convergence_checker.jl` | Convergence norm tracking |
| Utilities | `src/utilities/convergence_condition.jl` | Stopping criteria (absolute, relative) |
| Utilities | `src/utilities/line_search.jl` | Armijo line search |
| Utilities | `src/utilities/update_signal_handler.jl` | Jacobian update scheduling |
| Utilities | `src/utilities/async_utils.jl` | Async task helpers |
| Utilities | `src/utilities/sparse_coeffs.jl` | `SparseCoeffs` — compile-time zero-coefficient masking for tableaux |
| Utilities | `src/sparse_containers.jl` | `SparseContainer` — sparse coefficient containers for tableau storage |
| Callbacks | `src/Callbacks.jl` | Callback infrastructure (periodic, at-tstop) |
| Extensions | `ext/` | Package extensions (BenchmarkTools, CUDA profiling) |
| Tests | `test/` | See Test groups below |
| Benchmarks | `perf/` | Performance benchmarking scripts |
| Documentation | `docs/` | Documenter.jl setup; `docs/dev-guides/` is the DeveloperGuides subtree |

## Key abstractions

1. **`ClimaODEFunction`** (`src/functions.jl`) — the user-facing ODE function
   wrapper. Key keyword arguments:
   - `T_exp!(du, u, p, t)` — explicit tendency (not limited); increments `du`
   - `T_lim!(du, u, p, t)` — explicit tendency subject to the limiter; increments `du`
   - `T_exp_T_lim!(du_exp, du_lim, u, p, t)` — fused alternative to separate `T_exp!`/`T_lim!`
   - `T_imp!` — implicit tendency, typically a `ClimaTimeSteppers.ODEFunction` carrying `Wfact` and `jac_prototype`
   - `lim!(u, p, t, u_ref)` — monotonicity limiter applied after the limited explicit increment
   - `dss!(u, p, t)` — direct stiffness summation for spectral element continuity
   - `initialize_imp!(u, p, γdt)` — called once per implicit stage to set up the Newton solve
   - `cache!(u, p, t)` — update the parameter cache `p` to reflect state `u`
   - `cache_imp!(u, p, t)` — update cache components needed by `T_imp!` (defaults to `cache!`)

   Explicit-only, implicit-only, and IMEX problems are all expressed through this single type.

2. **`ODEProblem`** (`src/problems.jl`) — pairs a `ClimaODEFunction` with an
   initial state, time span, and parameters.

3. **`IMEXAlgorithm` / `ExplicitAlgorithm`** (`src/solvers/imex_tableaus.jl` and
   `src/solvers/explicit_tableaus.jl`) — top-level algorithm types.
   `IMEXAlgorithm` wraps an IMEX tableau name (e.g., `ARS343()`, `IMKG343a()`)
   and a `NewtonsMethod` instance. `ExplicitAlgorithm` wraps an explicit tableau
   name (e.g., `RK4()`, `LSRK54CarpenterKennedy()`).

4. **`NewtonsMethod`** (`src/nl_solvers/newtons_method.jl`) — configurable
   Newton solver. Key options:
   - `max_iters` — maximum Newton iterations (default `1`)
   - `krylov_method` — a `KrylovMethod` struct (GMRES or Jacobian-free) for
     the inner linear solve; `nothing` uses a direct `ldiv!`
   - `convergence_checker` — a `ConvergenceChecker` for early termination
   - `update_j` — an `UpdateSignalHandler` controlling when the Jacobian is
     refreshed (default: every Newton iteration)

   Users supply the Jacobian by wrapping `T_imp!` in a `ClimaTimeSteppers.ODEFunction`
   that carries `jac_prototype` (a pre-allocated operator) and `Wfact` (fills
   `W = dtγ J - I`).

5. **Tableau types** (`src/solvers/imex_tableaus.jl`,
   `src/solvers/explicit_tableaus.jl`) — pure-data structs describing Butcher
   tableaux. 30+ IMEX tableaux are provided (ARS, IMKG, SSP, ARK families up
   to 5th order). Tableau consistency is validated by
   `test/unit/tableaus.jl`.

## Test groups

Run the full suite with:
```julia
using Pkg; Pkg.test()
```
Or via the GitHub Actions workflow (`julia-actions/julia-runtest@v1`).

| Group | Test file(s) | What it covers |
|:---|:---|:---|
| Unit — sparse containers | `unit/sparse_containers.jl` | `SparseCoeffs` and sparse tableau storage |
| Unit — fused increment | `unit/fused_increment.jl` | `@fused_increment` correctness and allocation |
| Unit — tableau consistency | `unit/tableaus.jl` | Butcher tableau order conditions and consistency |
| Unit — update signal handler | `unit/update_signal_handler.jl` | Jacobian update scheduling logic |
| Unit — line search | `unit/line_search.jl` | Armijo line search correctness |
| Unit — convergence checker | `utils/convergence_checker.jl` | Norm tracking and stopping criteria |
| Unit — ForwardEuler | `unit/forward_euler.jl` | `ForwardEulerODEFunction` wrapper |
| Unit — incrementing ODE | `unit/incrementing_ode.jl` | `IncrementingODEFunction` wrapper |
| Unit — edge cases | `unit/edge_cases.jl` | Null explicit/implicit terms, degenerate inputs |
| Unit — Jacobian update | `unit/jacobian_update.jl` | Jacobian refresh strategies |
| Unit — Jacobian accuracy | `unit/jacobian_accuracy.jl` | Finite-difference Jacobian accuracy |
| Unit — ForwardDiff compat | `unit/forward_diff.jl` | Dual-number propagation through stepping |
| Solvers — Newton's method | `solvers/newtons_method.jl` | Newton/GMRES/JFNK correctness |
| Integration — ARS | `integration/single_column_ars.jl` | Single-column IMEX ARS end-to-end |
| Integration — callbacks | `integration/callbacks.jl` | Periodic and tstop callback execution |
| Integration — integrator | `integration/integrator.jl` | `init`/`step!`/`solve!` API coverage |
| Integration — limiter | `integration/limiter.jl` | Monotonicity limiter integration |
| Integration — SSP | `integration/ssp_monotonicity.jl` | SSP monotonicity guarantee |
| Integration — dense output | `integration/dense_output.jl` | Dense output interpolation |
| Integration — long-time | `integration/long_time_stability.jl` | Long-time stability and energy conservation |
| Type stability | `performance/type_stability.jl` | `@inferred` checks on stepping kernels |
| Step allocations | `performance/allocations.jl` | Zero-allocation stepping on CPU |
| Convergence — all solvers | `solvers/explicit_rk.jl`, `solvers/lsrk.jl`, `solvers/multirate.jl`, `solvers/rosenbrock.jl`, `solvers/imex_ark.jl` | Empirical convergence order for every solver family |
| Code quality | `aqua.jl` | Aqua.jl checks (ambiguities, unbound type params, stale deps) |

On Buildkite, the convergence tests run as separate parallel jobs (one per
solver file). Locally (`Pkg.test()`), they run sequentially inside one
`@safetestset`.

## Repo-specific conventions

- **`T_exp!` and `T_lim!` are incrementing**: Both explicit tendencies have
  signature `f!(du, u, p, t)` and must *add to* `du`, not overwrite it. The
  IMEX ARK stage loop accumulates Runge-Kutta increments by relying on this.
  `T_imp!` is wrapped in a `ClimaTimeSteppers.ODEFunction` and called
  differently (the Newton solver manages its own increment).
- **Zero-allocation stepping**: All stepping code under `src/solvers/` must be
  free of heap allocations on the hot path. Use `test/performance/allocations.jl`
  to guard regressions.
- **`Wfact` interface for implicit solves**: Pass `jac_prototype` (a
  pre-allocated operator) and `Wfact` (a function that fills `W = dtγ J - I`,
  where `dtγ` is the product of the timestep and the implicit diagonal
  coefficient, and `J` is the Jacobian of `T_imp!`) to the
  `ClimaTimeSteppers.ODEFunction` wrapping `T_imp!`. The Newton solver
  reads the operator via `jac_prototype` and refreshes it by calling `Wfact`.
- **Type stability is required**: All kernels must be `@inferred`-clean for the
  element type in use (usually `Float32` on GPU, `Float64` on CPU). Guard with
  `test/performance/type_stability.jl`.
- **Tableau additions**: New tableaux go in `src/solvers/imex_tableaus.jl` or
  `src/solvers/explicit_tableaus.jl`. Add order-condition tests to
  `test/unit/tableaus.jl` and a convergence test to the appropriate solver file
  under `test/solvers/`.
- **News entries**: Breaking changes and notable features must be recorded in
  `NEWS.md`.

## Self-correction

If this guide is discovered to be stale or missing a pattern, update it.
