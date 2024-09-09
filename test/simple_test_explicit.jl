import SciMLBase
import ClimaTimeSteppers

const FT = Float64

includet("../src/solvers/hard_coded_rosssp.jl")

function T_exp!(∂ₜY, Y, p, t)
    ∂ₜY .= p.λ * Y
    return nothing
end

Y_init = FT[1]
t_init = FT(0)
t_end = FT(10)
dt = FT(0.01)

prob = SciMLBase.ODEProblem(
    ClimaTimeSteppers.ClimaODEFunction(; T_exp!),
    Y_init,
    (t_init, t_end),
    (; λ = -1)
)

algo = RosSSPAlgorithm(tableau(DaiYagi()));

integrator = SciMLBase.init(prob, algo; dt, saveat = dt);

SciMLBase.solve!(integrator);
# SciMLBase.step!(integrator);
# SciMLBase.step!(integrator);
