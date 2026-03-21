using ClimaTimeSteppers
import ClimaTimeSteppers:
    ODEProblem, SplitODEProblem, IncrementingODEFunction, ODEFunction,
    SplitFunction, solve, init, solve!, step!, add_tstop!, reinit!,
    get_dt, set_dt!
import LinearAlgebra
import LinearAlgebra: norm, Diagonal, mul!, Tridiagonal, I

include(joinpath(@__DIR__, "problems", "linear_odes.jl"))
include(joinpath(@__DIR__, "problems", "imex_split_odes.jl"))
include(joinpath(@__DIR__, "utils", "test_case_utils.jl"))
include(joinpath(@__DIR__, "problems", "arkode_references.jl"))
include(joinpath(@__DIR__, "problems", "stiff_linear_ode.jl"))
include(joinpath(@__DIR__, "problems", "pdes.jl"))

# ============================================================================ #
# Test case collection
# ============================================================================ #

function all_test_cases(::Type{FT}) where {FT}
    return [
        ark_analytic_nonlin_test_cts(FT),
        ark_analytic_sys_test_cts(FT),
        ark_analytic_test_cts(FT),
        onewaycouple_mri_test_cts(FT),
        stiff_linear_test_cts(FT),
        finitediff_2Dheat_test_cts(FT),
        finitediff_1Dheat_test_cts(FT),
    ]
end
