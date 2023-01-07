# Algorithms

```@meta
CurrentModule = ClimaTimeSteppers
```

## Interface and OrdinaryDiffEq compatibility

```@docs
ForwardEulerODEFunction
```

## Generic IMEX methods

```@docs
IMEXTableaus
IMEXAlgorithm
```

## IMEX SSP methods

```@docs
SSP433
SSP222
SSP332
SSP333
SSP322
```

## Low-Storage Runge--Kutta (LSRK) methods

Low-storage Runger--Kutta methods reduce the number stages that need to be stored.
The methods below require only one additional storage vector.

An `IncrementingODEProblem` must be used.

```@docs
LowStorageRungeKutta2N
LSRK54CarpenterKennedy
LSRK144NiegemannDiehlBusch
LSRKEulerMethod
```

## Strong Stability-Preserving Runge--Kutta (SSPRK) methods

```@docs
StrongStabilityPreservingRungeKutta
SSPRK22Heuns
SSPRK22Ralstons
SSPRK33ShuOsher
SSPRK34SpiteriRuuth
```

## Additive Runge--Kutta (ARK) methods

ARK methods are IMEX (Implicit-Explicit) methods based on splitting the ODE function into a linear and remainder components:
```math
\frac{du}{dt} = Lu + f_R(u,t)
```
where the linear part is solved implicitly. All the algorithms below take a `linsolve` argument to specify the linear solver to be used.
See the [`linsolve` specification](https://diffeq.sciml.ai/latest/features/linear_nonlinear/) of DifferentialEquations.jl.

Currently ARK methods require a `SplitODEProblem`.

```@docs
AdditiveRungeKutta
ARK1ForwardBackwardEuler
ARK2ImplicitExplicitMidpoint
ARK2GiraldoKellyConstantinescu
ARK437L2SA1KennedyCarpenter
ARK548L2SA2KennedyCarpenter
```

### ARS

```@docs
ARS111
ARS121
ARS232
ARS343
```

## Multirate

```@docs
Multirate
```

### Multirate Infinitesimal Step

```@docs
MultirateInfinitesimalStep
MIS2
MIS3C
MIS4
MIS4a
TVDMISA
TVDMISB
```

### Wicker--Skamarock

```@docs
WickerSkamarockRungeKutta
WSRK2
WSRK3
```
