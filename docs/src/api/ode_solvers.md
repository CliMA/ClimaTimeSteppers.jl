# ODE Solvers

```@meta
CurrentModule = ClimaTimeSteppers
```

## Problem Types

```@docs
ODEProblem
SplitFunction
SplitODEProblem
IncrementingODEFunction
ODEFunction
ODESolution
```

## ODE Function Types

```@docs
ClimaODEFunction
ForwardEulerODEFunction
```

## Integrator

```@docs
init
solve
solve!
step!
reinit!
add_tstop!
get_dt
set_dt!
```

## Algorithm Constraints

```@docs
AbstractAlgorithmConstraint
Unconstrained
SSP
```

## IMEX Algorithms

```@docs
IMEXTableau
IMEXAlgorithm
ARS111
ARS121
ARS122
ARS233
ARS232
ARS222
ARS343
ARS443
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
SSP222
SSP322
SSP332
SSP333
SSP433
DBM453
HOMMEM1
ARK2GKC
ARK437L2SA1
ARK548L2SA2
SSPKnoth
```

## Explicit Algorithms

```@docs
ExplicitTableau
ExplicitAlgorithm
SSP22Heuns
SSP33ShuOsher
RK4
```

## Low-Storage Runge-Kutta

```@docs
ForwardEulerODEFunction
LowStorageRungeKutta2N
LSRK54CarpenterKennedy
LSRK144NiegemannDiehlBusch
LSRKEulerMethod
```

## Multirate

```@docs
Multirate
MultirateInfinitesimalStep
MIS2
MIS3C
MIS4
MIS4a
TVDMISA
TVDMISB
WickerSkamarockRungeKutta
WSRK2
WSRK3
```

## Rosenbrock

```@docs
RosenbrockAlgorithm
SSPKnoth
```
