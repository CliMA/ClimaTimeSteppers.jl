# ODE Solvers

```@meta
CurrentModule = ClimaTimeSteppers
```

This page documents all problem types, ODE function containers, integrator
functions, and time-stepping algorithms provided by ClimaTimeSteppers.jl.

For the mathematical derivations behind these algorithms, see the
[Algorithm Formulations](../algorithm_formulations/ode_solvers.md) section.

## Problem Types

These types define *what* to solve. `ODEProblem` is the main entry point;
`SplitODEProblem` is a convenience constructor for multirate problems.
Types not exported — access them qualified (e.g. `CTS.ODEProblem`).

```@docs
ODEProblem
SplitFunction
SplitODEProblem
IncrementingODEFunction
ODEFunction
ODESolution
```

## ODE Function Types

These types define *how* the right-hand side is structured. Most users will
use [`ClimaODEFunction`](@ref) for IMEX and Rosenbrock methods, or
[`IncrementingODEFunction`](@ref) for low-storage RK methods.

```@docs
ClimaODEFunction
ForwardEulerODEFunction
```

## Integrator

The integrator is the stateful object that advances the solution forward in
time. Create one with `init`, advance with `step!`, and run to completion
with `solve!`. The convenience function `solve` combines `init` and `solve!`.

```@docs
TimeStepperIntegrator
init
solve
solve!
step!
reinit!
add_tstop!
get_dt
set_dt!
```

## Abstract Types

```@docs
TimeSteppingAlgorithm
AbstractAlgorithmConstraint
Unconstrained
SSP
```

## IMEX Algorithms

Implicit-explicit additive Runge-Kutta methods. Use with
[`ClimaODEFunction`](@ref) for problems split into stiff (implicit) and
non-stiff (explicit) tendencies. See
[IMEX ARK formulation](../algorithm_formulations/ode_solvers.md) for the
mathematical details.

```@docs
IMEXTableau
IMEXAlgorithm
```

### ARS family ([ARS1997](@cite))
```@docs
ARS111
ARS121
ARS122
ARS233
ARS232
ARS222
ARS343
ARS443
```

### IMKG family ([SVTG2019](@cite))
```@docs
IMKG232a
IMKG232b
IMKG242a
IMKG242b
IMKG243a
IMKG252a
IMKG252b
IMKG253a
IMKG253b
IMKG254a
IMKG254b
IMKG254c
IMKG342a
IMKG343a
```

### SSP-IMEX family
```@docs
SSP222
SSP322
SSP332
SSP333
SSP433
```

### Other ARK methods
```@docs
DBM453
HOMMEM1
ARK2GKC
ARK437L2SA1
ARK548L2SA2
```

## Explicit Algorithms

Explicit Runge-Kutta methods for non-stiff problems.

```@docs
ExplicitTableau
ExplicitAlgorithm
SSP22Heuns
SSP33ShuOsher
RK4
```

## Low-Storage Runge-Kutta

Memory-efficient explicit methods using only two state-sized arrays.
Require an [`IncrementingODEFunction`](@ref). See the
[LSRK formulation](../algorithm_formulations/lsrk.md).

```@docs
LowStorageRungeKutta2N
LSRK54CarpenterKennedy
LSRK144NiegemannDiehlBusch
LSRKEulerMethod
```

## Multirate Methods

For problems with separated fast and slow timescales, using a
[`SplitODEProblem`](@ref). See the
[multirate formulation](../algorithm_formulations/mrrk.md).

```@docs
Multirate
```

### Multirate Infinitesimal Step (MIS)
```@docs
MultirateInfinitesimalStep
MIS2
MIS3C
MIS4
MIS4a
TVDMISA
TVDMISB
```

### Wicker-Skamarock
```@docs
WickerSkamarockRungeKutta
WSRK2
WSRK3
```

## Rosenbrock Methods

Linearly implicit methods that replace Newton iterations with a single
linear solve per stage. See the
[Rosenbrock formulation](../algorithm_formulations/rosenbrock.md).

```@docs
ClimaTimeSteppers.RosenbrockTableau
RosenbrockAlgorithm
SSPKnoth
```
