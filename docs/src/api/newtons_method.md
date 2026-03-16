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

## Jacobian Update Signals

The update signal system controls *when* the Jacobian is recomputed.
Freezing the Jacobian across iterations (chord method) or across timesteps
reduces cost at the expense of convergence rate.

```@docs
UpdateSignalHandler
UpdateEvery
UpdateEveryN
UpdateEveryDt
UpdateSignal
NewTimeStep
NewNewtonSolve
NewNewtonIteration
```

## Debugging

```@docs
KrylovMethodDebugger
PrintConditionNumber
```
