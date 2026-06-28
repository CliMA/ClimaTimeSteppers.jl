ClimaTimeSteppers.jl Release Notes
========================

main
-------

v0.10.1
-------
- ![][badge-💥breaking] Deprecated the standalone `SavedValues(tType, savevalType)` constructor; `SavedValues` is an internal container managed directly by `CTS.init`.
- ![][badge-💥breaking] `UpdateEveryDt(dt)` now takes the time-interval *value*, not an element
  type. (The previous `UpdateEveryDt(::Type)` form never produced a working handler.)
- ![][badge-✨feature/enhancement] `RosenbrockAlgorithm` is now exported and can be constructed
  from an algorithm name directly: `RosenbrockAlgorithm(SSPKnoth())`.
- ![][badge-✨feature/enhancement] `tableau(name)` now works for IMEX-ARK and explicit-RK names
  (e.g. `tableau(ARS343())`, `tableau(RK4())`), not only the low-storage / multirate families.
- ![][badge-🔥behavioralΔ] `EveryXSimulationTime` is now direction-aware and anchored to the start
  time, so it fires correctly under reverse-time integration and from a nonzero start time.
- ![][badge-🐛bugfix] Fixed wrong results from Wicker-Skamarock multirate methods: the inner
  integrator no longer overshoots its substep on state-coupled problems.
- ![][badge-🐛bugfix] The IMEX SSPRK stepper now honors the `initialize_imp!` hook (previously
  silently ignored on the SSP path).
- ![][badge-🐛bugfix] Fixed the iteration index passed to stateful `ConvergenceChecker` conditions
  (`MaximumErrorReduction`, `MinimumRateOfConvergence`) inside Newton's method.
- ![][badge-🐛bugfix] Fixed `UpdateEveryN`'s reset-signal dispatch (the counter previously never
  reset).
- ![][badge-🐛bugfix] `reinit!` now recomputes the integration direction, so a forward solve can be
  reinitialized as a reverse-time solve.
- ![][badge-🐛bugfix] Various robustness fixes: `RosenbrockAlgorithm` asserts a constant-diagonal Γ;
  `step!` with `advance_to_tstop` no longer errors when all tstops are consumed; out-of-range
  `saveat` values are filtered to `[t0, tf]`; `IMEXTableau`/`ExplicitTableau` validate stage
  counts and abscissae; `has_T_exp` is `false` for a limiter-only `ClimaODEFunction`.

v0.10.0
-------
- ![][badge-💥breaking] Removed `SciMLBase` and `DiffEqBase` backward-compatibility extensions
  (`ClimaTimeSteppersSciMLExt`, `ClimaTimeSteppersDiffEqExt`) and the `CTS.DiffEqBase` stub module.
  Use `CTS.init`, `CTS.solve`, `CTS.step!`, etc. directly.
- ![][badge-💥breaking] Removed backward-compatibility aliases `DistributedODEAlgorithm`
  (use `TimeSteppingAlgorithm`) and `DistributedODEIntegrator` (use `TimeStepperIntegrator`).
  Removed the `continuous_callbacks` field from `CallbackSet`; use only `discrete_callbacks`.
- ![][badge-✨feature/enhancement] Added `constrain_state!` hook on `ClimaODEFunction`
  for enforcing physical state constraints. Applied after `dss!`.
- ![][badge-✨feature/enhancement] Added an update-signal hierarchy
  `EndOfStep <: EndOfStage <: WithDSS <: UpdateSignal`, and new keyword
  arguments `update_cache` and `update_constrain_state` on `ClimaODEFunction`
  that accept any `UpdateSignalHandler` to control when `cache!` and
  `constrain_state!` fire. Defaults: `UpdateEvery(EndOfStage)` for `cache!`
  and `UpdateEvery(EndOfStep)` for `constrain_state!`. `UpdateEvery(WithDSS)`
  fires at every `dss!` site (including pre-implicit and post-`initialize_imp!`
  DSSes).
- ![][badge-✨feature/enhancement] Added `is_fsal(tableau)` / `is_fsal(alg)`
  trait for IMEX tableaux (`b == a[end, :]` on both explicit and implicit
  sides). The IMEX-ARK integrator uses this to skip the redundant post-Newton
  `cache!` / `constrain_state!` firing at end of the final stage when
  `u ≡ U_s`, since the end-of-step firing already handles the same state.

v0.8.9
------
- Major refactor of CTS, including removal of SciML and DiffEq dependency (maintained in extension for backward compatibility)
- Refactor and extension of test suite
- Correction of bugs in multirate and Krylov solvers

v0.8.7
------
- Added a simple line search algorithm to Newtons method.

v0.8.5
------
- Removed SciMLBase` type `restriction on `step_u!`.

v0.8.2
------
- ![][badge-💥breaking] If saveat is a number, then it does not automatically expand to `tspan[1]:saveat:tspan[2]`. To fix this, update
`saveat`, which is a keyword in the integrator, to be an array. For example, if `saveat` is a scalar, replace it with
`[tspan[1]:saveat:tspan[2]..., tspan[2]]` to achieve the same behavior as before.
- IMEXAlgorithms and SSPKnoth are compatible with ITime. See ClimaUtilities for more information about ITime.

v0.7.18
-------
- ![][badge-🚀performance] Added a fused `T_exp_T_lim!`, so that we can fuse DSS calls that previously lived in `T_exp!` and `T_lim!`.

v0.7.17
-------
- Started NEWS
- ![][badge-🤖precisionΔ]![][badge-🚀performance] increments are now fused in the imex ARK method. This should yield better GPU performance, and may result in machine-precision changes.

<!--
Contributors are welcome to begin the description of changelog items with badge(s) below. Here is a brief description of when to use badges for a particular pull request / set of changes:
 - 🔥behavioralΔ - behavioral changes. For example: a new model is used, yielding more accurate results.
 - 🤖precisionΔ - machine-precision changes. For example, swapping the order of summed arguments can result in machine-precision changes.
 - 💥breaking - breaking changes. For example: removing deprecated functions/types, removing support for functionality, API changes.
 - 🚀performance - performance improvements. For example: improving type inference, reducing allocations, or code hoisting.
 - ✨feature - new feature added. For example: adding support for a cubed-sphere grid
 - 🐛bugfix - bugfix. For example: fixing incorrect logic, resulting in incorrect results, or fixing code that otherwise might give a `MethodError`.
-->

[badge-🔥behavioralΔ]: https://img.shields.io/badge/🔥behavioralΔ-orange.svg
[badge-🤖precisionΔ]: https://img.shields.io/badge/🤖precisionΔ-black.svg
[badge-💥breaking]: https://img.shields.io/badge/💥BREAKING-red.svg
[badge-🚀performance]: https://img.shields.io/badge/🚀performance-green.svg
[badge-✨feature/enhancement]: https://img.shields.io/badge/feature/enhancement-blue.svg
[badge-🐛bugfix]: https://img.shields.io/badge/🐛bugfix-purple.svg
