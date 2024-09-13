import SciMLBase
import ClimaTimeSteppers

const FT = Float64

function T_imp!(∂ₜY, Y, p, t)
    ∂ₜY .= p.λ1 * Y
    return nothing
end

function T_exp!(∂ₜY, Y, p, t)
    ∂ₜY .= p.λ2 * Y
    return nothing
end

function Wfact!(W, Y, p, dtγ, t)
    W .= dtγ * p.λ1 - 1
    return nothing
end

T_imp_wrapper! = SciMLBase.ODEFunction(T_imp!; jac_prototype = zeros(1, 1), Wfact = Wfact!);

dts = [0.1, 0.01, 0.001]

errors = map(dts) do dt
    Y_init = FT[1]
    t_init = FT(0)
    t_end = FT(10)
    dt = FT(dt)
    p = (; λ1 = -1/2, λ2 = -1/3)

    prob = SciMLBase.ODEProblem(
        ClimaTimeSteppers.ClimaODEFunction(; T_imp! = T_imp_wrapper!, T_exp!),
        Y_init,
        (t_init, t_end),
        p
    )

    algo = ClimaTimeSteppers.RosSSPScalarAlgorithm(ClimaTimeSteppers.tableau(ClimaTimeSteppers.DaiYagiScalar()));

    integrator = SciMLBase.init(prob, algo; dt, saveat = dt);
    SciMLBase.solve!(integrator);
    return abs(integrator.sol.u[end][end] - exp((p.λ1 + p.λ2) * t_end))
end

@show errors

log_dts = log10.(dts)
log_errs = log10.(errors)
n_dts = length(dts)

order, _ = hcat(log_dts, ones(n_dts)) \ log_errs
@show order

# SciMLBase.step!(integrator);
# SciMLBase.step!(integrator);
