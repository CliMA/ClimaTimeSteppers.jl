# ClimaTimeSteppers.jl

ClimaTimeSteppers.jl is a suite of ordinary differential equation (ODE) solvers for use as
time-stepping methods in a partial differential equation (PDE) solver, such as
[ClimateMachine.jl](https://github.com/CliMA/ClimateMachine.jl). They are specifically
written to support distributed and GPU computation, while minimising the memory footprint.

ClimaTimeSteppers.jl provides self-contained ODE problem types, solvers, and a callback
system, with optional backward compatibility with the
[DifferentialEquations.jl ecosystem](https://diffeq.sciml.ai/latest/) via a package extension.

```@docs
ClimaTimeSteppers
```
