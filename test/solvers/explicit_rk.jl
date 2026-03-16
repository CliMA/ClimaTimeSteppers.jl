#=
julia --project=test
using Revise; include("test/solvers/explicit_rk.jl")
=#
using ClimaTimeSteppers, LinearAlgebra, Test, DiffEqBase

if !@isdefined(IntegratorTestCase)
    include(joinpath(@__DIR__, "..", "utils", "convergence_utils.jl"))
end

function explicit_linear_test_cts(::Type{FT}) where {FT}
    prob = ODEProblem(
        ClimaODEFunction(; T_exp! = (du, u, p, t) -> (du .= p .* u)),
        FT[1 / 2],
        (FT(0.0), FT(1.0)),
        FT(1.01),
    )
    return IntegratorTestCase(
        "Explicit Linear (T_exp)",
        false,
        FT(1.0),
        (t) -> linear_sol(prob.u0, prob.p, t),
        prob,
        prob,
        200,
        1,
    )
end

function explicit_sincos_test_cts(::Type{FT}) where {FT}
    prob = ODEProblem(
        ClimaODEFunction(;
            T_exp! = (du, u, p, t) -> (du[1] = p * u[2]; du[2] = -p * u[1]; nothing),
        ),
        FT[0.0, 1.0],
        (FT(0.0), FT(1.0)),
        FT(2.0),
    )
    return IntegratorTestCase(
        "Explicit SinCos (T_exp)",
        false,
        FT(1.0),
        (t) -> sincos_sol(prob.u0, prob.p, t),
        prob,
        prob,
        200,
        1,
    )
end

EXPLICIT_ALGORITHMS = (
    SSP22Heuns,
    SSP33ShuOsher,
    RK4,
)

@testset "Explicit RK Algorithms Convergence" begin
    # IMEX SSP methods run in purely explicit mode for these tests.
    algorithm(algorithm_name::ClimaTimeSteppers.IMEXSSPRKAlgorithmName, linear_implicit) =
        ExplicitAlgorithm(algorithm_name)

    for alg in EXPLICIT_ALGORITHMS
        alg_name = alg()
        @testset "$(nameof(alg))" begin
            test_convergence!(alg_name, explicit_linear_test_cts(Float64), 100)
            test_convergence!(alg_name, explicit_sincos_test_cts(Float64), 800)

            # Float32 convergence: rounding degrades observed order for low-order
            # methods. SSP22Heuns consistently fails all Float32 tests.
            test_convergence!(alg_name, explicit_linear_test_cts(Float32), 10;
                num_steps_scaling_factor = 2,
                broken_tests = (SSP22Heuns(),))
            test_convergence!(alg_name, explicit_sincos_test_cts(Float32), 10;
                num_steps_scaling_factor = 2,
                broken_tests = (SSP22Heuns(), SSP33ShuOsher()))
        end
    end
end
