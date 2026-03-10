using DiffEqBase, ClimaTimeSteppers
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
        # TODO: Enable once convergence test infrastructure supports per-problem
        # dt ranges. The λ=1000 stiffness requires much smaller dt values than
        # the current refinement_range 5:9 (dt ≈ 0.03–0.002).
        # stiff_linear_test_cts(FT),
        finitediff_2Dheat_test_cts(FT),
        finitediff_1Dheat_test_cts(FT),
    ]
end
