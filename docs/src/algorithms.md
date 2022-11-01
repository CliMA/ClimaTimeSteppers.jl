# Algorithms

```@meta
CurrentModule = ClimaTimeSteppers
```

## Interface and OrdinaryDiffEq compatibility

```@docs
allocate_cache
run!
ForwardEulerODEFunction
```

## IMEX ARK methods

```@docs
IMEXARKAlgorithm
make_IMEXARKAlgorithm
```

The convergence orders of the provided methods are verified using test cases from [ARKode](http://runge.math.smu.edu/ARKode_example.pdf). Plots of the solutions to these test cases, the errors of these solutions, and the convergence orders with respect to `dt` are shown below.

```@setup plots
using Pkg
Pkg.activate("../../test")
Pkg.instantiate()
include("../../test/problems.jl")
include("../../test/utils.jl")
include("../../test/convergence.jl")
Pkg.activate(".")
```

Plots for `ark_analytic`:
![](output/solutions_ark_analytic_imex_ark.png)
![](output/errors_ark_analytic_imex_ark.png)
![](output/orders_ark_analytic_imex_ark.png)

Plots for `ark_analytic_nonlin`:
![](output/solutions_ark_analytic_nonlin_imex_ark.png)
![](output/errors_ark_analytic_nonlin_imex_ark.png)
![](output/orders_ark_analytic_nonlin_imex_ark.png)

Plots for `ark_analytic_sys`:
![](output/solutions_ark_analytic_sys_imex_ark.png)
![](output/errors_ark_analytic_sys_imex_ark.png)
![](output/orders_ark_analytic_sys_imex_ark.png)

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
