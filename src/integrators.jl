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
    integrator = DistributedODEIntegrator(prob, alg, u, dt, t, tstop, 0, stepstop, adjustfinal, callbackset, false, cache(prob, alg, dt))

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
        DiffEqBase.step!(integrator)

        if integrator.step == integrator.stepstop
            break
        end
    end    
end


# either called directly, or via solve!
function DiffEqBase.step!(integrator::DistributedODEIntegrator)
    step_u!(integrator, integrator.cache) # solvers need to define this interface
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

function adjust_dt!(integrator::DistributedODEIntegrator, dt)
    # TODO: figure out interface for recomputing other objects (linear operators, etc)
    integrator.dt = dt
end


# not sure what this should do?
# defined as default initialize: https://github.com/SciML/DiffEqBase.jl/blob/master/src/callbacks.jl#L3
DiffEqBase.u_modified!(i::DistributedODEIntegrator,bool) = nothing
