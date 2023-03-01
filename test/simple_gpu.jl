using ClimaTimeSteppers, LinearAlgebra, Test, CUDA

include(joinpath(@__DIR__, "convergence_orders.jl"))
include(joinpath(@__DIR__, "convergence_utils.jl"))
include(joinpath(@__DIR__, "utils.jl"))
include(joinpath(@__DIR__, "problems.jl"))


@testset "LSRK and SSP convergence" begin
    dts = 0.5 .^ (4:7)

    for (prob, sol, tscale) in [
        (linear_prob(CuArray), linear_sol, 1)
        (sincos_prob(CuArray), sincos_sol, 1)
    ]

        @test convergence_order(prob, sol, LSRKEulerMethod(), dts .* tscale) ≈ 1 atol = 0.1
        @test convergence_order(prob, sol, LSRK54CarpenterKennedy(), dts .* tscale) ≈ 4 atol = 0.05
        @test convergence_order(prob, sol, LSRK144NiegemannDiehlBusch(), dts .* tscale) ≈ 4 atol = 0.05

        @test convergence_order(prob, sol, SSPRK22Heuns(), dts .* tscale) ≈ 2 atol = 0.05
        @test convergence_order(prob, sol, SSPRK22Ralstons(), dts .* tscale) ≈ 2 atol = 0.05
        @test convergence_order(prob, sol, SSPRK33ShuOsher(), dts .* tscale) ≈ 3 atol = 0.05
        @test convergence_order(prob, sol, SSPRK34SpiteriRuuth(), dts .* tscale) ≈ 3 atol = 0.05
    end
end
