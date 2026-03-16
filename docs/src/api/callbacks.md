# Callbacks

```@meta
CurrentModule = ClimaTimeSteppers
```

Callbacks allow user code to run at specific points during time integration —
after every step, at fixed simulation-time intervals, or on a wall-clock
schedule. They are passed to [`init`](@ref) or [`solve`](@ref) via the
`callback` keyword argument.

## Core Types

[`DiscreteCallback`](@ref) is the primitive: a condition/action pair checked
after every integrator step. [`CallbackSet`](@ref) groups multiple callbacks.

```@docs
DiscreteCallback
CallbackSet
```

## Pre-Built Callbacks

```@meta
CurrentModule = ClimaTimeSteppers.Callbacks
```

The `ClimaTimeSteppers.Callbacks` module provides convenience constructors for
common triggering patterns. Import them with:

```julia
using ClimaTimeSteppers: Callbacks
using .Callbacks
```

```@docs
Callbacks
EveryXWallTimeSeconds
EveryXSimulationTime
EveryXSimulationSteps
```

## Extension Points

Define methods on `initialize!` and `finalize!` for your callback's `f!`
type to run setup/teardown code when the integrator starts and finishes.

```@docs
initialize!
finalize!
```
