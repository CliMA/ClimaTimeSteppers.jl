# Algorithms

## Low-Storage Runge--Kutta (LSRK) methods

Low-storage Runger--Kutta methods reduce the number stages that need to be stored.
The methods below require only one additional storage vector.

An `IncrementingODEProblem` must be used.

```@docs
LSRK54CarpenterKennedy
LSRK144NiegemannDiehlBusch
LSRKEulerMethod
```

## Strong Stability-Preserving Runge--Kutta (SSPRK) methods


```@docs
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
ARK1ForwardBackwardEuler
ARK2ImplicitExplicitMidpoint
ARK2GiraldoKellyConstantinescu
ARK437L2SA1KennedyCarpenter
ARK548L2SA2KennedyCarpenter
```

## Multirate Infinitesimal Step

```@docs
MIS2
MIS3C
MIS4
MIS4a
TVDMISA
TVDMISB
```

