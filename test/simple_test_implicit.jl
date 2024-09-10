import SciMLBase
import ClimaTimeSteppers

const FT = Float64

includet("../src/solvers/hard_coded_rosssp.jl")

function T_imp!(∂ₜY, Y, p, t)
    ∂ₜY .= p.λ * Y
    return nothing
end

function Wfact!(W, Y, p, dtγ, t)
    W .= dtγ * p.λ - 1
    return nothing
end

T_imp_wrapper! = SciMLBase.ODEFunction(T_imp!; jac_prototype = zeros(1, 1), Wfact = Wfact!);

Y_init = FT[1]
t_init = FT(0)
t_end = FT(10)
dt = FT(0.001)

prob = SciMLBase.ODEProblem(
    ClimaTimeSteppers.ClimaODEFunction(; T_imp! = T_imp_wrapper!),
    Y_init,
    (t_init, t_end),
    (; λ = -1/2)
)

algo = RosSSPAlgorithm(tableau(DaiYagi()));

integrator = SciMLBase.init(prob, algo; dt, saveat = dt);

SciMLBase.solve!(integrator);
integrator.sol.u[end][end]
# SciMLBase.step!(integrator);
# SciMLBase.step!(integrator);
