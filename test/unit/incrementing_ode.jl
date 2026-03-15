#=
Unit tests for IncrementingODEFunction.
=#
using ClimaTimeSteppers, DiffEqBase, LinearAlgebra, Test

@testset "IncrementingODEFunction" begin

    @testset "Basic α/β semantics" begin
        # IncrementingODEFunction{true} expects: du = α * f(u) + β * du
        f! = IncrementingODEFunction{true}(
            (du, u, p, t, α = true, β = false) -> (du .= α .* p .* u .+ β .* du),
        )
        u = [2.0, 4.0]
        du = [10.0, 20.0]
        p = -0.5
        t = 0.0

        # Default: α=true (=1), β=false (=0) → du = 1 * (-0.5) * u + 0 * du
        f!(du, u, p, t)
        @test du ≈ [-1.0, -2.0]

        # With α=2, β=0 → du = 2 * (-0.5) * u
        du .= 999.0
        f!(du, u, p, t, 2.0, false)
        @test du ≈ [-2.0, -4.0]

        # With α=1, β=1 → du = (-0.5) * u + du (accumulate)
        du .= [10.0, 20.0]
        f!(du, u, p, t, 1.0, 1.0)
        @test du ≈ [10.0 + (-1.0), 20.0 + (-2.0)]
    end

    @testset "Float32 precision" begin
        f! = IncrementingODEFunction{true}(
            (du, u, p, t, α = true, β = false) -> (du .= α .* p .* u .+ β .* du),
        )
        u = Float32[1.0, 2.0]
        du = similar(u)
        f!(du, u, Float32(-1.0), Float32(0.0))
        @test eltype(du) == Float32
        @test du ≈ Float32[-1.0, -2.0]
    end

    @testset "Multi-step convergence (forward Euler via LSRK)" begin
        # du/dt = -u → u(t) = exp(-t)
        # Using LSRKEulerMethod which is a 1-stage LSRK (forward Euler)
        f! = IncrementingODEFunction{true}(
            (du, u, p, t, α = true, β = false) -> (du .= α .* p .* u .+ β .* du),
        )
        prob = ODEProblem(f!, [1.0], (0.0, 1.0), -1.0)
        hide = (; kwargshandle = DiffEqBase.KeywordArgSilent)
        sol = solve(prob, LSRKEulerMethod(); dt = 0.001, save_everystep = false, hide...)
        @test sol.u[end][1] ≈ exp(-1.0) atol = 0.01
    end
end
