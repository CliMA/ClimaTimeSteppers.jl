"""
    DistributedODEIntegrator <: AbstractODEIntegrator

A simplified variant of [`ODEIntegrator`](https://github.com/SciML/OrdinaryDiffEq.jl/blob/6ec5a55bda26efae596bf99bea1a1d729636f412/src/integrators/type.jl#L77-L123).

"""
mutable struct DistributedODEIntegrator{algType,uType,tType} <: DiffEqBase.AbstractODEIntegrator{algType,true,uType,tType}
    prob
    alg::algType
    u::uType
    dt::tType
    t::tType
    tstop::tType
    step::Int
    stepstop::Int # -1
    adjustfinal::Bool
    callback::DiffEqBase.CallbackSet
    u_modified::Bool
    cache
end

# called by DiffEqBase.init and solve (see below)
function DiffEqBase.__init(
    prob::DiffEqBase.AbstractODEProblem,
    alg::DistributedODEAlgorithm,
    args...;
    dt,  # required
    stepstop=-1,
    adjustfinal=false,
    callback=nothing,
    kwargs...)

    u = prob.u0
    t = prob.tspan[1]
    tstop = prob.tspan[2]

    callbackset = DiffEqBase.CallbackSet(callback)
    isempty(callbackset.continuous_callbacks) || error("Continuous callbacks are not supported")
    integrator = DistributedODEIntegrator(prob, alg, u, dt, t, tstop, 0, stepstop, adjustfinal, callbackset, false, init_cache(prob, alg; dt=dt, kwargs...))

    DiffEqBase.initialize!(callbackset,u,t,integrator)
    return integrator
end


# called by DiffEqBase.solve
function DiffEqBase.__solve(
    prob::DiffEqBase.AbstractODEProblem,
    alg::DistributedODEAlgorithm,
    args...;
    kwargs...)

    integrator = DiffEqBase.__init(prob, alg, args...; kwargs...)
    DiffEqBase.solve!(integrator)
    return integrator.u # ODEProblem returns a Solution objec
end

# either called directly (after init), or by solv e (via __solve)
function DiffEqBase.solve!(integrator::DistributedODEIntegrator)
    while integrator.t < integrator.tstop
        if integrator.adjustfinal && integrator.t + integrator.dt > integrator.tstop
            adjust_dt!(integrator, integrator.tstop - integrator.t)
        end
        if !integrator.adjustfinal && integrator.t + integrator.dt/2 > integrator.tstop
            break
        end

        DiffEqBase.step!(integrator)

        if integrator.step == integrator.stepstop
            break
        end
    end

    if isdefined(DiffEqBase, :finalize!)
        DiffEqBase.finalize!(integrator.callback, integrator.u, integrator.t, integrator)
    end
end


# either called directly, or via solve!
function DiffEqBase.step!(integrator::DistributedODEIntegrator)
    step_u!(integrator) # solvers need to define this interface
    integrator.t += integrator.dt
    integrator.step += 1

    # apply callbacks
    discrete_callbacks = integrator.callback.discrete_callbacks
    for callback in discrete_callbacks
        if callback.condition(integrator.u,integrator.t,integrator)
            callback.affect!(integrator)
        end
    end
end

# solvers need to define this interface
step_u!(integrator) = step_u!(integrator, integrator.cache)

"""
    adjust_dt!(integrator::DistributedODEIntegrator, dt[, dt_cache=nothing])

Adjust the time step of the integrator to `dt`. The optional `dt_cache` object
can be passed when the integrator has a `dt`-dependent component that needs to
be updated (such as a linear solver).
"""
function adjust_dt!(integrator::DistributedODEIntegrator, dt, dt_cache=nothing)
    # TODO: figure out interface for recomputing other objects (linear operators, etc)
    integrator.dt = dt
    adjust_dt!(integrator.cache, dt, dt_cache)
end

# interfaces

"""
    init_cache(prob, alg::A; kwargs...)::AC

Construct an algorithm cache for the algorithm `alg`. This should be defined
for any algorithm type `A`, and should return an object of an appropriate cache
type `AC` that can be dispatched on for [`step_u!`](@ref) and/or
[`init_inner`](@ref)/[`update_inner!`](@ref).
"""
function init_cache end

"""
    step_u!(integrator, cache::AC)

Perform a single step that updates the state `integrator.u` using accordint to
the algorithm corresponding to `cache`.

This should be defined for any algorithm cache type `AC` that can be used
directly or as an inner timestepper. For outer timesteppers,
[`init_inner`](@ref) and [`update_inner!`](@ref) need to be defined instead.
"""
step_u!(integrator, cache)

"""
    init_dt_cache(cache::AC, prob, dt)

Construct a `dt`-dependent subcache of `cache` for the ODE problem `prob`. This
should _not_ modify `cache` itself, but return an object that can be passed as
the `dt_cache` argument to [`adjust_dt!`](@ref).

By default this returns `nothing`. This should be defined for any algorithm
cache type `AC` which has `dt`-dependent components.

For example, an implicit solver can use this to return a factorized Euler
operator ``I-dt*L`` that is used as part of the implicit solve.

This initialization will typically be done as part of [`init_cache`](@ref)
itself: this interface is provided for multirate schemes which need to modify
the `dt` of the inner solver at each outer stage.
"""
function init_dt_cache(cache, prob, dt)
    return nothing
end


function get_dt_cache(cache)
    return nothing
end

"""
    adjust_dt!(cache::AC, dt, dt_cache)

Adjust the time step of the algorithm cache `cache`. This should be defined for
any algorithm cache type `AC`, where `dt_cache` is an object returned by
[`init_dt_cache`](@ref).
"""
adjust_dt!(cache, dt, dt_cache)



# not sure what this should do?
# defined as default initialize: https://github.com/SciML/DiffEqBase.jl/blob/master/src/callbacks.jl#L3
DiffEqBase.u_modified!(i::DistributedODEIntegrator,bool) = nothing
