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

# Re-export DiffEqBase through CTS for downstream access (e.g., CTS.DiffEqBase.KeywordArgSilent)
if !isdefined(CTS, :DiffEqBase)
    @eval CTS const DiffEqBase = $DiffEqBase
end


# NOTE: SciMLBase.ODEProblem(f::CTS.ClimaODEFunction, ...) is defined in
# ClimaTimeSteppersSciMLExt (which is always loaded when this extension loads,
# since DiffEqBase depends on SciMLBase). Do NOT duplicate it here.


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

# NOTE: solve!, step!, add_tstop!, get_dt, has_reinit, reinit!, u_modified!,
# allows_arbitrary_number_types, and allowscomplex are all defined in
# ClimaTimeSteppersSciMLExt. DiffEqBase re-exports these from SciMLBase,
# so defining them here would overwrite the SciMLExt methods.

end # module
