#=
Unit tests for the OffsetODEFunction wrapper. It evaluates a wrapped function
`f` at the offset time `α + β*t` and adds a constant forcing `γ*x`, supporting
3-arg (out-of-place), 4-arg, 5-arg (`α`), and 6-arg (`α, β`) call forms (see
src/functions.jl). Multirate solvers mutate the scalar fields between stages.
=#
using ClimaTimeSteppers, Test
import ClimaTimeSteppers: OffsetODEFunction

# Wrapped function f(u, t) = u .* t, with one method per call form that the
# OffsetODEFunction dispatches to. `a`, `b` are the stage coefficients passed
# to the 5/6-arg forms (named `a`/`b` to avoid clashing with the offset's α/β).
ftest(u, p, t) = u .* t                                    # 3-arg out-of-place
ftest(du, u, p, t) = (du .= u .* t)                        # 4-arg in-place
ftest(du, u, p, t, a) = (du .= a .* u .* t)                # 5-arg
ftest(du, u, p, t, a, b) = (du .= a .* u .* t .+ b .* du)  # 6-arg

@testset "OffsetODEFunction" begin
    α, β, γ, x = 1.0, 2.0, -1.0, [0.5, 1.0]
    u, p, t = [2.0, 4.0], nothing, 0.5
    t_eff = α + β * t   # 1.0 + 2.0 * 0.5 = 2.0
    offset = OffsetODEFunction(ftest, α, β, γ, x)

    @testset "3-arg (out-of-place)" begin
        # f(u, t_eff) .+ γ.*x = [4,8] .+ [-0.5,-1] = [3.5, 7.0]
        @test offset(u, p, t) ≈ [3.5, 7.0]
    end

    @testset "4-arg (in-place)" begin
        du = similar(u)
        offset(du, u, p, t)
        @test du ≈ [3.5, 7.0]
    end

    @testset "5-arg (in-place, α)" begin
        a = 3.0
        du = similar(u)
        # f(du, u, t_eff, a) = a.*u.*t_eff = [12,24]; du .+= a.*γ.*x = [-1.5,-3]
        offset(du, u, p, t, a)
        @test du ≈ [10.5, 21.0]
    end

    @testset "6-arg (in-place, α, β)" begin
        a, b = 3.0, 0.5
        du = [1.0, 1.0]   # prior value, accumulated via the b.*du term
        # f(du, u, t_eff, a, b) = a.*u.*t_eff .+ b.*du = [12,24] .+ [0.5,0.5]
        #                       = [12.5, 24.5]; du .+= a.*γ.*x = [-1.5,-3]
        offset(du, u, p, t, a, b)
        @test du ≈ [11.0, 21.5]
    end

    @testset "fields are mutable (used by multirate solvers)" begin
        offset.α = 0.0
        offset.β = 1.0
        offset.γ = 0.0
        # t_eff = 0.5, f(u, t_eff) = [1.0, 2.0], no forcing
        du = similar(u)
        offset(du, u, p, t)
        @test du ≈ [1.0, 2.0]
    end
end
