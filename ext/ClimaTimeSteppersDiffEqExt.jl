"""
Backward compatibility extension for DiffEqBase/SciMLBase.

This extension ensures that downstream packages (ClimaAtmos, ClimaLand, ClimaCoupler)
that call `DiffEqBase.init(prob, alg; ...)`, `DiffEqBase.solve!(integrator)`, etc.
continue to work during the transition period.
"""
module ClimaTimeSteppersDiffEqExt

import DiffEqBase
import SciMLBase
import ClimaTimeSteppers as CTS

# Convert DiffEqBase function types to CTS equivalents
convert_f(f) = f
convert_f(f::SciMLBase.SplitFunction) = CTS.SplitFunction(f.f1, f.f2)

# Forward DiffEqBase.init → CTS.init
function DiffEqBase.__init(
    prob::DiffEqBase.AbstractODEProblem,
    alg::CTS.DistributedODEAlgorithm,
    args...;
    kwargs...,
)
    # Convert DiffEqBase.ODEProblem to local ODEProblem
    local_prob = CTS.ODEProblem(convert_f(prob.f), prob.u0, prob.tspan, prob.p)
    CTS.init(local_prob, alg, args...; kwargs...)
end

# Also accept ClimaTimeSteppers' own ODEProblem through DiffEqBase.init
function DiffEqBase.__init(
    prob::CTS.ODEProblem,
    alg::CTS.DistributedODEAlgorithm,
    args...;
    kwargs...,
)
    CTS.init(prob, alg, args...; kwargs...)
end

# Forward DiffEqBase.solve → CTS.solve
function DiffEqBase.__solve(
    prob::DiffEqBase.AbstractODEProblem,
    alg::CTS.DistributedODEAlgorithm,
    args...;
    kwargs...,
)
    local_prob = CTS.ODEProblem(convert_f(prob.f), prob.u0, prob.tspan, prob.p)
    CTS.solve(local_prob, alg, args...; kwargs...)
end

function DiffEqBase.__solve(
    prob::CTS.ODEProblem,
    alg::CTS.DistributedODEAlgorithm,
    args...;
    kwargs...,
)
    CTS.solve(prob, alg, args...; kwargs...)
end

# Forward DiffEqBase.solve! → CTS.solve!
DiffEqBase.solve!(integrator::CTS.DistributedODEIntegrator) = CTS.solve!(integrator)

# Forward DiffEqBase.step! → CTS.step!
DiffEqBase.step!(integrator::CTS.DistributedODEIntegrator) = CTS.step!(integrator)
DiffEqBase.step!(integrator::CTS.DistributedODEIntegrator, dt, stop_at_tdt = false) =
    CTS.step!(integrator, dt, stop_at_tdt)

# Forward DiffEqBase.add_tstop! → CTS.add_tstop!
DiffEqBase.add_tstop!(integrator::CTS.DistributedODEIntegrator, t) =
    CTS.add_tstop!(integrator, t)

# Forward DiffEqBase.get_dt
DiffEqBase.get_dt(integrator::CTS.DistributedODEIntegrator) = CTS.get_dt(integrator)

# Forward DiffEqBase.reinit!
DiffEqBase.has_reinit(::CTS.DistributedODEIntegrator) = true
function DiffEqBase.reinit!(integrator::CTS.DistributedODEIntegrator, args...; kwargs...)
    CTS.reinit!(integrator, args...; kwargs...)
end

# u_modified! is a DiffEqBase-only concept (no-op here)
DiffEqBase.u_modified!(::CTS.DistributedODEIntegrator, ::Bool) = nothing

# SciMLBase: allows_arbitrary_number_types
SciMLBase.allows_arbitrary_number_types(::CTS.DistributedODEAlgorithm) = true
SciMLBase.allowscomplex(::CTS.DistributedODEAlgorithm) = true

end # module
