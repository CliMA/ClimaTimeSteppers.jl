"""
    ClimaTimeSteppers

ODE solvers for climate model time-stepping. Designed for distributed and GPU
computation with minimal memory footprint.

# Core workflow

```julia
import ClimaTimeSteppers as CTS

# Define tendency functions and Jacobian (W = dtγ J - I) for the implicit part
T_imp = CTS.ODEFunction(T_imp!; jac_prototype = W, Wfact = Wfact!)
f    = CTS.ClimaODEFunction(; T_exp! = T_exp!, T_imp! = T_imp)
prob = CTS.ODEProblem(f, u0, tspan, p)
alg  = IMEXAlgorithm(ARS343(), NewtonsMethod(; max_iters = 2))
sol  = CTS.solve(prob, alg; dt = 0.01)
```

Or step-by-step:

```julia
integrator = CTS.init(prob, alg; dt = 0.01)
CTS.step!(integrator)        # one step
CTS.solve!(integrator)       # run to completion
```

# Key types
- [`ODEProblem`](@ref): problem definition (function, initial state, time span, parameters)
- [`ClimaODEFunction`](@ref): tendency container for IMEX / Rosenbrock solvers
- [`TimeSteppingAlgorithm`](@ref): abstract supertype for all algorithms
- [`TimeStepperIntegrator`](@ref): mutable integrator state
- [`ODESolution`](@ref): solution container with saved time points and states
"""
module ClimaTimeSteppers


using LinearAlgebra
using LinearOperators
using StaticArrays
import ClimaComms
using NVTX

export AbstractAlgorithmName, AbstractAlgorithmConstraint, Unconstrained, SSP

# Note: init, solve, solve!, step!, add_tstop!, reinit!, get_dt, set_dt!,
# ODEProblem, ODEFunction, SplitODEProblem, IncrementingODEFunction
# are intentionally NOT exported yet to avoid conflicts with SciMLBase.
# Access them qualified, e.g.: CTS.init, CTS.ODEProblem, CTS.solve!, etc.

import LinearAlgebra, Krylov

include(joinpath("utilities", "sparse_coeffs.jl"))
include(joinpath("utilities", "fused_increment.jl"))
include("sparse_containers.jl")
include("problems.jl")
include("functions.jl")

"""
    TimeSteppingAlgorithm

Abstract supertype for all ODE algorithms in ClimaTimeSteppers.

Concrete subtypes include [`IMEXAlgorithm`](@ref),
[`LowStorageRungeKutta2N`](@ref), [`Multirate`](@ref), and
[`RosenbrockAlgorithm`](@ref).
"""
abstract type TimeSteppingAlgorithm end
"""Backward-compatible alias for [`TimeSteppingAlgorithm`](@ref)."""
const DistributedODEAlgorithm = TimeSteppingAlgorithm

abstract type AbstractAlgorithmName end

"""
    AbstractAlgorithmConstraint

A mechanism for constraining which operations can be performed by an algorithm
for solving ODEs.

For example, an unconstrained algorithm might compute a Runge-Kutta stage by
taking linear combinations of tendencies; i.e., by adding quantities of the form
`dt * tendency(state)`. On the other hand, a "strong stability preserving"
algorithm can only take linear combinations of "incremented states"; i.e., it
only adds quantities of the form `state + dt * coefficient * tendency(state)`.
"""
abstract type AbstractAlgorithmConstraint end

"""
    Unconstrained

Indicates that an algorithm may perform any supported operations.
"""
struct Unconstrained <: AbstractAlgorithmConstraint end

default_constraint(::AbstractAlgorithmName) = Unconstrained()

"""
    SSP

Indicates that an algorithm must be "strong stability preserving", which makes
it easier to guarantee that the algorithm will preserve monotonicity properties
satisfied by the initial state. For example, this ensures that the algorithm
will be able to use limiters in a mathematically consistent way.
"""
struct SSP <: AbstractAlgorithmConstraint end

"""
    DiscreteCallback(condition, affect!; initialize, finalize)

A callback checked after each integrator step.

# Arguments
- `condition`: function `(u, t, integrator) -> Bool`
- `affect!`: function `(integrator) -> nothing`, called when `condition` returns `true`

# Keyword Arguments
- `initialize`: function `(cb, u, t, integrator)` called at integrator startup
- `finalize`: function `(cb, u, t, integrator)` called when [`solve!`](@ref) finishes

See also [`CallbackSet`](@ref), [`ClimaTimeSteppers.Callbacks`](@ref).
"""
struct DiscreteCallback{C, A, I, F}
    condition::C
    affect!::A
    initialize::I
    finalize::F
end
function DiscreteCallback(
    condition, affect!;
    initialize = (cb, u, t, integrator) -> nothing,
    finalize = (cb, u, t, integrator) -> nothing,
)
    DiscreteCallback(condition, affect!, initialize, finalize)
end

"""
    CallbackSet(callbacks...)

A collection of [`DiscreteCallback`](@ref)s applied after each step.
Accepts `nothing`, individual `DiscreteCallback`s, or nested `CallbackSet`s.
"""
struct CallbackSet{DC <: Tuple}
    discrete_callbacks::DC
end
CallbackSet(cbs::DiscreteCallback...) = CallbackSet(cbs)
function CallbackSet(cb_or_nothing, cbs::DiscreteCallback...)
    if isnothing(cb_or_nothing)
        CallbackSet(cbs)
    elseif cb_or_nothing isa DiscreteCallback
        CallbackSet((cb_or_nothing, cbs...))
    elseif cb_or_nothing isa CallbackSet
        CallbackSet((cb_or_nothing.discrete_callbacks..., cbs...))
    else
        error("Unsupported callback type: $(typeof(cb_or_nothing))")
    end
end

function initialize_callbacks!(cbset::CallbackSet, u, t, integrator)
    for cb in cbset.discrete_callbacks
        cb.initialize(cb, u, t, integrator)
    end
end
function finalize_callbacks!(cbset::CallbackSet, u, t, integrator)
    for cb in cbset.discrete_callbacks
        cb.finalize(cb, u, t, integrator)
    end
end

include("integrators.jl")
"""Backward-compatible alias for `TimeStepperIntegrator`."""
const DistributedODEIntegrator = TimeStepperIntegrator

include("utilities/update_signal_handler.jl")
include("utilities/convergence_condition.jl")
include("utilities/convergence_checker.jl")
include("utilities/line_search.jl")
include("nl_solvers/newtons_method.jl")


n_stages_ntuple(::Type{<:NTuple{Nstages}}) where {Nstages} = Nstages
n_stages_ntuple(::Type{<:SVector{Nstages}}) where {Nstages} = Nstages

# Include concrete implementations
const SPCO = SparseCoeffs

include("solvers/imex_tableaus.jl")
include("solvers/explicit_tableaus.jl")
include("solvers/imex_ark.jl")
include("solvers/imex_ssprk.jl")
include("solvers/multirate.jl")
include("solvers/lsrk.jl")
include("solvers/mis.jl")
include("solvers/wickerskamarock.jl")
include("solvers/rosenbrock.jl")

include("Callbacks.jl")


benchmark_step(integrator, device) =
    @warn "Must load CUDA, BenchmarkTools, OrderedCollections, StatsBase, PrettyTables to trigger the ClimaTimeSteppersBenchmarkToolsExt extension"

end
