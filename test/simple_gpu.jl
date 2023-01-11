using ClimaTimeSteppers
import ClimaTimeSteppers as CTS
import OrdinaryDiffEq as ODE

include(joinpath(@__DIR__, "problems.jl"))

function main(::Type{FT}) where {FT}
    alg_name = ARS343()
    test_case = climacore_2Dheat_test_cts(FT; print_arr_type = true)
    prob = test_case.split_prob
    alg = CTS.IMEXAlgorithm(alg_name, NewtonsMethod(; max_iters = 2))
    integrator = ODE.init(prob, alg; dt = FT(0.01))
    sol = ODE.solve!(integrator)
    @info "Done!"
    return integrator
end

integrator = main(Float64)
nothing
