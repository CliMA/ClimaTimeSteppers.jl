#=
julia --project=test
using Revise; include("test/solvers/explicit_rk.jl")
=#
using ClimaTimeSteppers, LinearAlgebra, Test, DiffEqBase

if !@isdefined(IntegratorTestCase)
    include(joinpath(@__DIR__, "..", "utils", "convergence_utils.jl"))
end

# Explicit RK uses ExplicitAlgorithm which wraps IMEXAlgorithm, so it needs
# ClimaODEFunction problems (not IncrementingODEFunction).

function explicit_linear_prob()
    ODEProblem(
        ClimaODEFunction(; T_exp! = (du, u, p, t) -> (du .= p .* u)),
        [1 / 2],
        (0.0, 1.0),
        1.01,
    )
end

function explicit_sincos_prob()
    ODEProblem(
        ClimaODEFunction(; T_exp! = (du, u, p, t) -> (du[1] = p * u[2]; du[2] = -p * u[1])),
        [0.0, 1.0],
        (0.0, 1.0),
        2.0,
    )
end

@testset "Explicit RK convergence" begin
    dts = 0.5 .^ (4:7)

    for (prob, sol, tscale, name) in [
        (explicit_linear_prob(), linear_sol, 1, "linear")
        (explicit_sincos_prob(), sincos_sol, 1, "sincos")
    ]
        @testset "$name" begin
            @testset "SSP22Heuns" begin
                @test convergence_order(
                    prob,
                    sol,
                    ExplicitAlgorithm(SSP22Heuns()),
                    dts .* tscale,
                ) ≈ 2 atol = 0.1
            end

            @testset "SSP33ShuOsher" begin
                @test convergence_order(
                    prob,
                    sol,
                    ExplicitAlgorithm(SSP33ShuOsher()),
                    dts .* tscale,
                ) ≈ 3 atol = 0.1
            end

            @testset "RK4" begin
                @test convergence_order(
                    prob,
                    sol,
                    ExplicitAlgorithm(RK4()),
                    dts .* tscale,
                ) ≈ 4 atol = 0.05
            end
        end
    end
end

# Float32 convergence: verify that Float32 state vectors achieve same convergence orders.
# Use Float64 tspan/params since DiffEqBase requires consistent time types.
function explicit_linear_prob_f32()
    ODEProblem(
        ClimaODEFunction(; T_exp! = (du, u, p, t) -> (du .= p .* u)),
        Float32[1 / 2],
        (0.0, 1.0),
        1.01,
    )
end

@testset "Explicit RK Float32 convergence" begin
    # Use coarser dts for Float32 to stay above Float32 precision floor.
    # Fine dts produce truncation errors below eps(Float32) ≈ 1.2e-7,
    # causing the convergence study to break down.
    # For high-order methods (order ≥ 3), Float32 rounding degrades the
    # observed convergence order significantly (e.g., RK4 measures ~3.6
    # instead of 4), so we only check order > 2.
    dts_f32_low = 0.5 .^ (2:5)   # for order ≤ 2
    dts_f32_high = 0.5 .^ (1:4)  # for order ≥ 3
    prob = explicit_linear_prob_f32()
    @test convergence_order(
        prob,
        linear_sol_f32,
        ExplicitAlgorithm(SSP22Heuns()),
        dts_f32_low,
    ) ≈ 2 atol = 0.2
    @test convergence_order(
        prob,
        linear_sol_f32,
        ExplicitAlgorithm(SSP33ShuOsher()),
        dts_f32_high,
    ) > 2
    @test convergence_order(prob, linear_sol_f32, ExplicitAlgorithm(RK4()), dts_f32_high) >
          2
end
