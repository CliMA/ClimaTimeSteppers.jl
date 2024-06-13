# ODE Solvers

```@meta
CurrentModule = ClimaTimeSteppers
```

## Tableau Interface

```@docs
ClimaTimeSteppers.RKTableau
ClimaTimeSteppers.ButcherTableau
ClimaTimeSteppers.ShuOsherTableau
ClimaTimeSteppers.PaddedTableau
ClimaTimeSteppers.ARKTableau
ClimaTimeSteppers.is_ERK
ClimaTimeSteppers.is_DIRK
```

## Algorithm Interface

```@docs
AbstractAlgorithmName
ClimaTimeSteppers.RKAlgorithmName
ClimaTimeSteppers.SSPRKAlgorithmName
ClimaTimeSteppers.ARKAlgorithmName
ClimaTimeSteppers.IMEXSSPRKAlgorithmName
RKAlgorithm
ARKAlgorithm
```

## RK Algorithm Names

```@docs
SSP22Heuns
SSP33ShuOsher
RK4
```

## ARK Algorithm Names

```@docs
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

## Old LSRK Interface

```@docs
ForwardEulerODEFunction
LowStorageRungeKutta2N
LSRK54CarpenterKennedy
LSRK144NiegemannDiehlBusch
LSRKEulerMethod
```

## Old Multirate Interface

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
