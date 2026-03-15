#=
julia --project=test
using Revise; include("test/solvers/imex_ark.jl")
=#
using ClimaTimeSteppers, LinearAlgebra, Test
import SciMLBase

if !@isdefined(IntegratorTestCase)
    include(joinpath(@__DIR__, "..", "utils", "convergence_utils.jl"))
end

ALGORITHMS = (
    ARS111,
    ARS121,
    ARS122,
    ARS232,
    ARS222,
    IMKG232a,
    IMKG232b,
    IMKG242a,
    IMKG242b,
    IMKG243a,
    IMKG252a,
    IMKG252b,
    IMKG253a,
    IMKG253b,
    IMKG254a,
    IMKG254b,
    IMKG254c,
    HOMMEM1,
    ARS233,
    ARS343,
    ARS443,
    IMKG342a,
    IMKG343a,
    DBM453,
    ARK2GKC,
    ARK437L2SA1,
    ARK548L2SA2,
    SSP222,
    SSP322,
    SSP332,
    SSP333,
    SSP433,
)

@testset "IMEX ARK Algorithms Convergence" begin
    for alg in ALGORITHMS
        alg_name = alg()
        @testset "$(nameof(alg))" begin
            test_convergence!(alg_name, ark_analytic_nonlin_test_cts(Float64), 450)
            test_convergence!(alg_name, ark_analytic_sys_test_cts(Float64), 200)
            test_convergence!(
                alg_name,
                ark_analytic_test_cts(Float64),
                25000;
                super_convergence_algorithm_names = (ARS121(),),
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

            if alg_name isa ClimaTimeSteppers.IMEXSSPRKAlgorithmName
                test_unconstrained_vs_ssp_without_limiters(
                    alg_name,
                    ark_analytic_nonlin_test_cts(Float64),
                    450,
                )
                test_unconstrained_vs_ssp_without_limiters(
                    alg_name,
                    ark_analytic_sys_test_cts(Float64),
                    200,
                )
                test_unconstrained_vs_ssp_without_limiters(
                    alg_name,
                    ark_analytic_test_cts(Float64),
                    25000,
                )
                test_unconstrained_vs_ssp_without_limiters(
                    alg_name,
                    finitediff_1Dheat_test_cts(Float64),
                    200,
                )
                test_unconstrained_vs_ssp_without_limiters(
                    alg_name,
                    finitediff_2Dheat_test_cts(Float64),
                    200,
                )
                test_unconstrained_vs_ssp_without_limiters(
                    alg_name,
                    onewaycouple_mri_test_cts(Float64),
                    9000,
                )
            end
        end
    end
end
