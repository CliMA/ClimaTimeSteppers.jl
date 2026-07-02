# Newton's Method

```@meta
CurrentModule = ClimaTimeSteppers
```

This page documents the nonlinear solver infrastructure used by the implicit
stages of IMEX and Rosenbrock methods. For the mathematical background, see
the [Newton's method formulation](../algorithm_formulations/newtons_method.md).

## Newton Solver

[`NewtonsMethod`](@ref) is the top-level type passed to [`IMEXAlgorithm`](@ref).
It controls how many iterations to run, when to recompute the Jacobian,
whether to use a Krylov linear solver, and optional convergence checking
and line search.

```@docs
NewtonsMethod
LineSearch
```

## Krylov Linear Solver

When the Jacobian is too expensive to form or factor, a Krylov method
(e.g. GMRES) can solve the linear system approximately. The accuracy is
controlled by a *forcing term* that sets the Krylov tolerance.

```@docs
KrylovMethod
ForcingTerm
ConstantForcing
EisenstatWalkerForcing
```

## Jacobian-Free Newton-Krylov (JFNK)

When combined with a Jacobian-free JVP, the Newton-Krylov method never
forms or stores the Jacobian matrix — it only needs the action ``Jv``,
approximated via finite differences.

```@docs
JacobianFreeJVP
ForwardDiffJVP
ForwardDiffStepSize
ForwardDiffStepSize1
ForwardDiffStepSize2
ForwardDiffStepSize3
```

## Convergence Checking

These types control when Newton's method should terminate early (before
`max_iters`) based on the residual or update norm.

```@docs
ConvergenceChecker
ConvergenceCondition
MaximumError
MaximumRelativeError
MaximumErrorReduction
MinimumRateOfConvergence
MultipleConditions
```

## Update Signals

The update signal system controls *when* several hooks are re-invoked
during a step. Handlers ([`UpdateEvery`](@ref), [`UpdateEveryN`](@ref),
[`UpdateEveryDt`](@ref)) subscribe to a signal *type*; dispatch is by
subtype match, so a broader signal type fires on any of its subtype
signals.

The Jacobian update policy (`update_j`) uses the Newton-specific signals
(`NewTimeStep` / `NewNewtonSolve` / `NewNewtonIteration`); freezing the
Jacobian across iterations (chord method) or across timesteps reduces
cost at the expense of convergence rate.

The `cache!` / `constrain_state!` update policies
(`update_cache` / `update_constrain_state` on [`ClimaODEFunction`](@ref))
use the DSS-related signal hierarchy:

- [`WithDSS`](@ref) — every `dss!` call site (pre-implicit DSS,
  post-`initialize_imp!` DSS, post-Newton DSS, and end of step).
- [`EndOfStage`](@ref) — state-ready-for-tendency-evaluation moments
  (post-Newton in implicit stages, right after DSS in explicit-only
  stages, and end of step). Subtype of `WithDSS`.
- [`EndOfStep`](@ref) — end of a full time step. Subtype of
  `EndOfStage`.

So `UpdateEvery(WithDSS)` fires on every DSS site; `UpdateEvery(EndOfStage)`
fires only on state-ready moments; `UpdateEvery(EndOfStep)` fires only at
end of step. On FSAL tableaux (see [`is_fsal`](@ref)), the redundant
post-Newton fire at end of stage `s` is skipped because `u ≡ U_s`
already.

```@docs
UpdateSignalHandler
UpdateEvery
UpdateEveryN
UpdateEveryDt
UpdateSignal
NewTimeStep
NewNewtonSolve
NewNewtonIteration
WithDSS
EndOfStage
EndOfStep
```

## Debugging

```@docs
KrylovMethodDebugger
PrintConditionNumber
```
