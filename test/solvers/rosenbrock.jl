#=
julia --project=test
using Revise; include("test/solvers/rosenbrock.jl")
=#
using ClimaTimeSteppers, LinearAlgebra, Test

if !@isdefined(IntegratorTestCase)
    include(joinpath(@__DIR__, "..", "utils", "convergence_utils.jl"))
end

ROSENBROCK_ALGORITHMS = (
    SSPKnoth,
)

@testset "Rosenbrock Algorithms Convergence" begin
    for alg in ROSENBROCK_ALGORITHMS
        alg_name = alg()
        @testset "$(nameof(alg))" begin
            test_convergence!(alg_name, ark_analytic_nonlin_test_cts(Float64), 450)
            test_convergence!(alg_name, ark_analytic_sys_test_cts(Float64), 200)
            test_convergence!(
                alg_name,
                ark_analytic_test_cts(Float64),
                25000;
            )
            test_convergence!(
                alg_name,
                onewaycouple_mri_test_cts(Float64),
                9000;
                high_order_sample_shifts = 3,
            )
            test_convergence!(
                alg_name,
                stiff_linear_test_cts(Float64),
                1000;
                numerical_reference_algorithm_name = ARK548L2SA2(),
            )
            test_convergence!(
                alg_name,
                finitediff_1Dheat_test_cts(Float64),
                200;
                numerical_reference_algorithm_name = ARK548L2SA2(),
            )
            test_convergence!(
                alg_name,
                finitediff_2Dheat_test_cts(Float64),
                200;
                numerical_reference_algorithm_name = ARK548L2SA2(),
            )
        end
    end
end
