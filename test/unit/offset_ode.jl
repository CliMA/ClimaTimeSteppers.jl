#=
Unit tests for OffsetODEFunction wrapper.
=#
using ClimaTimeSteppers, Test
import ClimaTimeSteppers: OffsetODEFunction

@testset "OffsetODEFunction" begin
    # True function: du = u * (α + β*t) + γ*x
    # Signature supports out-of-place and in-place. We will test in-place.
    f! = (du, u, p, t) -> (du .= u .* t)

    α_val = 1.0
    β_val = 2.0
    γ_val = -1.0
    x_val = [0.5, 1.0]

    offset_f! = OffsetODEFunction(f!, α_val, β_val, γ_val, x_val)

    u = [2.0, 4.0]
    p = nothing
    t = 0.5

    # Expected: du = u * (α + β*t) + γ*x
    # t_eff = 1.0 + 2.0 * 0.5 = 2.0
    # f(u, p, t_eff) = u * 2.0 = [4.0, 8.0]
    # γ*x = -1.0 * [0.5, 1.0] = [-0.5, -1.0]
    # du_expected = [3.5, 7.0]

    du = similar(u)

    # Test standard 4-arg in-place signature
    offset_f!(du, u, p, t)
    @test du ≈ [3.5, 7.0]

    # Test that mutating the fields updates the output dynamically (used by multirate solver)
    offset_f!.α = 0.0
    offset_f!.β = 1.0
    offset_f!.γ = 0.0

    # Now t_eff = 0.0 + 1.0 * 0.5 = 0.5
    # f(u, p, t_eff) = [1.0, 2.0]
    # γ*x = 0
    offset_f!(du, u, p, t)
    @test du ≈ [1.0, 2.0]
end
