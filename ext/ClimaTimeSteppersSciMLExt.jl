"""
SciMLBase-only backward compatibility extension.

Provides full backward compatibility for downstream packages that use
SciMLBase.ODEProblem, SciMLBase.init, SciMLBase.solve!, SciMLBase.step!, etc.
with CTS algorithms and integrators — without requiring DiffEqBase.
"""
module ClimaTimeSteppersSciMLExt

import SciMLBase
import ClimaTimeSteppers as CTS

# ---------------------------------------------------------------------------
# ODEProblem construction
# ---------------------------------------------------------------------------
# SciMLBase.ODEProblem(f, u0, tspan, p) normally calls isinplace(f, 4) which
# uses numargs(f)/methods(f) introspection. ClimaODEFunction is not directly
# callable with (du, u, p, t), so this check fails. We bypass it.
function SciMLBase.ODEProblem(
    f::CTS.ClimaODEFunction,
    u0,
    tspan,
    p = SciMLBase.NullParameters();
    kwargs...,
)
    _tspan = SciMLBase.promote_tspan(tspan)
    _f = SciMLBase.ODEFunction{true, SciMLBase.FullSpecialize}(f)
    SciMLBase.ODEProblem{true}(_f, u0, _tspan, p; kwargs...)
end

# ---------------------------------------------------------------------------
# init / solve forwarding (direct dispatch, not via __init/__solve hooks)
# ---------------------------------------------------------------------------
# Without DiffEqBase loaded, SciMLBase.init has no method for AbstractDEProblem.
# DiffEqBase.init(::AbstractDEProblem, ...) → init_up → __init is the normal chain,
# but without DiffEqBase we must define init directly.
convert_f(f) = f
convert_f(f::SciMLBase.SplitFunction) = CTS.SplitFunction(f.f1, f.f2)
# Unwrap SciMLBase.ODEFunction to extract the inner ClimaODEFunction
convert_f(f::SciMLBase.ODEFunction) = f.f

# Convert SciMLBase callback types to CTS equivalents
convert_cb(cb) = cb   # pass through CTS callbacks and nothing unchanged
function convert_cb(cb::SciMLBase.DiscreteCallback)
    CTS.DiscreteCallback(
        cb.condition, cb.affect!;
        initialize = cb.initialize,
        finalize = cb.finalize,
    )
end
function convert_cb(cbs::SciMLBase.CallbackSet)
    # SciMLBase.CallbackSet has .continuous_callbacks and .discrete_callbacks
    converted = map(convert_cb, cbs.discrete_callbacks)
    CTS.CallbackSet(converted)
end

# Convert the callback kwarg if present
function convert_kwargs(kwargs)
    pairs = Dict{Symbol, Any}(kwargs)
    if haskey(pairs, :callback)
        pairs[:callback] = convert_cb(pairs[:callback])
    end
    return pairs
end

# SciMLBase.init → CTS.init
function SciMLBase.init(
    prob::SciMLBase.AbstractODEProblem,
    alg::CTS.DistributedODEAlgorithm,
    args...;
    kwargs...,
)
    local_prob = CTS.ODEProblem(convert_f(prob.f), prob.u0, prob.tspan, prob.p)
    CTS.init(local_prob, alg, args...; convert_kwargs(kwargs)...)
end

function SciMLBase.init(
    prob::CTS.ODEProblem,
    alg::CTS.DistributedODEAlgorithm,
    args...;
    kwargs...,
)
    CTS.init(prob, alg, args...; convert_kwargs(kwargs)...)
end

# SciMLBase.solve → CTS.solve
function SciMLBase.solve(
    prob::SciMLBase.AbstractODEProblem,
    alg::CTS.DistributedODEAlgorithm,
    args...;
    kwargs...,
)
    local_prob = CTS.ODEProblem(convert_f(prob.f), prob.u0, prob.tspan, prob.p)
    CTS.solve(local_prob, alg, args...; kwargs...)
end

function SciMLBase.solve(
    prob::CTS.ODEProblem,
    alg::CTS.DistributedODEAlgorithm,
    args...;
    kwargs...,
)
    CTS.solve(prob, alg, args...; kwargs...)
end

# ---------------------------------------------------------------------------
# solve! / step! / add_tstop! / get_dt / reinit! / u_modified!
# ---------------------------------------------------------------------------
SciMLBase.solve!(integrator::CTS.DistributedODEIntegrator) = CTS.solve!(integrator)

SciMLBase.step!(integrator::CTS.DistributedODEIntegrator) = CTS.step!(integrator)
SciMLBase.step!(integrator::CTS.DistributedODEIntegrator, dt, stop_at_tdt = false) =
    CTS.step!(integrator, dt, stop_at_tdt)

SciMLBase.add_tstop!(integrator::CTS.DistributedODEIntegrator, t) =
    CTS.add_tstop!(integrator, t)

SciMLBase.get_dt(integrator::CTS.DistributedODEIntegrator) = CTS.get_dt(integrator)

SciMLBase.has_reinit(::CTS.DistributedODEIntegrator) = true
function SciMLBase.reinit!(integrator::CTS.DistributedODEIntegrator, args...; kwargs...)
    CTS.reinit!(integrator, args...; kwargs...)
end

SciMLBase.u_modified!(::CTS.DistributedODEIntegrator, ::Bool) = nothing

# ---------------------------------------------------------------------------
# Callback lifecycle for SciMLBase.CallbackSet (ClimaDiagnostics wraps in SciMLBase types)
# ---------------------------------------------------------------------------
function CTS.initialize_callbacks!(cbset::SciMLBase.CallbackSet, u, t, integrator)
    for cb in cbset.discrete_callbacks
        cb.initialize(cb, u, t, integrator)
    end
end
function CTS.finalize_callbacks!(cbset::SciMLBase.CallbackSet, u, t, integrator)
    for cb in cbset.discrete_callbacks
        cb.finalize(cb, u, t, integrator)
    end
end

# SciMLBase: allows_arbitrary_number_types
SciMLBase.allows_arbitrary_number_types(::CTS.DistributedODEAlgorithm) = true
SciMLBase.allowscomplex(::CTS.DistributedODEAlgorithm) = true

end # module
