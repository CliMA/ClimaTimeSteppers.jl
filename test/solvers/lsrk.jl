#=
julia --project=test
using Revise; include("test/solvers/lsrk.jl")
=#
using ClimaTimeSteppers, LinearAlgebra, Test
import PrettyTables
import SciMLBase

if !@isdefined(IntegratorTestCase)
    include(joinpath(@__DIR__, "..", "utils", "convergence_utils.jl"))
end

@testset "LSRK convergence" begin
    dts = 0.5 .^ (4:7)

    for (prob, sol, tscale) in [
        (linear_prob(), linear_sol, 1)
        (sincos_prob(), sincos_sol, 1)
    ]

        @test convergence_order(prob, sol, LSRKEulerMethod(), dts .* tscale) ≈ 1 atol = 0.1
        @test convergence_order(prob, sol, LSRK54CarpenterKennedy(), dts .* tscale) ≈ 4 atol =
            0.05
        @test convergence_order(prob, sol, LSRK144NiegemannDiehlBusch(), dts .* tscale) ≈ 4 atol =
            0.05
    end
end

# Float32 convergence for LSRK: Float32 state vectors with Float64 tspan.
function linear_prob_f32()
    ODEProblem(
        IncrementingODEFunction{true}(
            (du, u, p, t, α = true, β = false) -> (du .= α .* p .* u .+ β .* du),
        ),
        Float32[1 / 2],
        (0.0, 1.0),
        1.01,
    )
end

@testset "LSRK Float32 convergence" begin
    # Use coarser dts for Float32 (see explicit_rk.jl for rationale).
    # For order-4 LSRK methods, Float32 rounding degrades the observed
    # convergence order (e.g., LSRK144 measures ~2.5 instead of 4),
    # so we only check order > 2 for high-order methods.
    dts_f32 = 0.5 .^ (1:4)
    prob = linear_prob_f32()
    @test convergence_order(prob, linear_sol_f32, LSRKEulerMethod(), dts_f32) ≈ 1 atol = 0.2
    @test convergence_order(prob, linear_sol_f32, LSRK54CarpenterKennedy(), dts_f32) > 2
    @test convergence_order(prob, linear_sol_f32, LSRK144NiegemannDiehlBusch(), dts_f32) > 2
end
