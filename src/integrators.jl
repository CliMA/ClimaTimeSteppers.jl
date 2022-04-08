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
    tdir::tType
    step::Int
    stepstop::Int # -1
    adjustfinal::Bool
    callback::DiffEqBase.CallbackSet
    u_modified::Bool
    cache
    sol
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
    save_func=(u,t,integrator)->copy(u),
    saveat=nothing,
    save_everystep=false,
    kwargs...)

    u = prob.u0
    t = prob.tspan[1]
    tstop = prob.tspan[2]
    tdir = sign(tstop - t)

    sol = DiffEqBase.build_solution(prob, alg, typeof(t)[], typeof(u)[])
    if isnothing(saveat)
        saveat = [tstop]
    elseif saveat isa Number
        saveat = t:saveat:tstop
    end
    saving_callback = NonInterpolatingSavingCallback(save_func, DiffEqCallbacks.SavedValues(sol.t, sol.u); saveat, save_everystep)
    callbackset = DiffEqBase.CallbackSet(callback, saving_callback)
    isempty(callbackset.continuous_callbacks) || error("Continuous callbacks are not supported")
    integrator = DistributedODEIntegrator(prob, alg, u, dt, t, tstop, tdir, 0, stepstop, adjustfinal, callbackset, false, cache(prob, alg; dt=dt, kwargs...), sol)
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
end

# either called directly (after init), or by solve (via __solve)
function DiffEqBase.solve!(integrator::DistributedODEIntegrator)
    while integrator.t < integrator.tstop
        if integrator.adjustfinal && integrator.t + integrator.dt > integrator.tstop
            adjust_dt!(integrator, integrator.tstop - integrator.t)
        end
        DiffEqBase.step!(integrator)

        if integrator.step == integrator.stepstop
            break
        end
    end

    if isdefined(DiffEqBase, :finalize!)
        DiffEqBase.finalize!(integrator.callback, integrator.u, integrator.t, integrator)
    end
    if isempty(integrator.sol.t)
        push!(integrator.sol.t, integrator.t)
        push!(integrator.sol.u, integrator.u)
    end
    return integrator.sol
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

function adjust_dt!(integrator::DistributedODEIntegrator, dt)
    # TODO: figure out interface for recomputing other objects (linear operators, etc)
    integrator.dt = dt
end


# not sure what this should do?
# defined as default initialize: https://github.com/SciML/DiffEqBase.jl/blob/master/src/callbacks.jl#L3
DiffEqBase.u_modified!(i::DistributedODEIntegrator,bool) = nothing


# this is roughly based on SavingCallback from DiffEqCallbacks, except that it
# doesn't interpolate; instead it will save the first step after
function NonInterpolatingSavingCallback(save_func, saved_values::DiffEqCallbacks.SavedValues;
    saveat=nothing,
    save_everystep=false,
)
    if isnothing(saveat) && save_everystep == false
        error("saveat or save_everystep must be defined")
    end
    if save_everystep
        condition = (u, t, integrator) -> true
    else
        saveat = collect(saveat)
        condition = (u, t, integrator) -> begin
            cond = false
            while !isempty(saveat) && first(saveat) <= t
                cond = true
                popfirst!(saveat)
            end
            return cond
        end
    end
    function affect!(integrator)
        push!(saved_values.t, integrator.t)
        push!(saved_values.saveval, save_func(integrator.u, integrator.t, integrator))
    end
    initialize(cb, u, t, integrator) = condition(u,t,integrator) && affect!(integrator)
    finalize(cb, u, t, integrator) = !save_everystep && !isempty(saveat) && affect!(integrator)
    DiffEqBase.DiscreteCallback(condition, affect!; initialize=initialize, finalize=finalize)
end