import SciMLBase
import ClimaTimeSteppers

const FT = Float64

function T_exp!(∂ₜY, Y, p, t)
    ∂ₜY .= p.λ * Y
    return nothing
end

Y_init = FT[1]
t_init = FT(0)
t_end = FT(10)
dt = FT(0.001)

prob = SciMLBase.ODEProblem(
    ClimaTimeSteppers.ClimaODEFunction(; T_exp!),
    Y_init,
    (t_init, t_end),
    (; λ = -1)
)

algo = ClimaTimeSteppers.RosSSPAlgorithm(ClimaTimeSteppers.tableau(ClimaTimeSteppers.DaiYagi()));

integrator = SciMLBase.init(prob, algo; dt, saveat = dt);

SciMLBase.solve!(integrator);
# SciMLBase.step!(integrator);
# SciMLBase.step!(integrator);
