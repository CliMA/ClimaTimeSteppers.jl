#=
Solution-lookup tests for `(sol::ODESolution)(t)`.

`sol(t)` is a NEAREST-NEIGHBOR lookup, NOT an interpolation: it returns
`sol.u[argmin(abs.(sol.t .- t))]` (see src/problems.jl). These tests pin down
that contract — exact match at saved points, nearest-neighbor selection between
them, tie-breaking, and clamping for out-of-range queries.
=#
using ClimaTimeSteppers, LinearAlgebra, Test
import ClimaTimeSteppers as CTS
import ClimaTimeSteppers: ODEProblem, ODEFunction, ODESolution, solve

@testset "Solution lookup sol(t)" begin

    @testset "Explicit method: sol(t) matches at saved points" begin
        prob = ODEProblem(
            ClimaODEFunction(; T_exp! = (du, u, p, t) -> (du .= -u)),
            [1.0, 2.0],
            (0.0, 1.0),
            nothing,
        )
        alg = ExplicitAlgorithm(SSP33ShuOsher())
        sol = solve(prob, alg; dt = 0.01, save_everystep = true)

        # Querying at a saved time returns exactly that saved state (the stored
        # object, not a recomputed/interpolated value).
        for (i, t) in enumerate(sol.t)
            @test sol(t) == sol.u[i]
        end
    end

    @testset "IMEX method: sol(t) matches at saved points" begin
        n = 2
        Id = Matrix{Float64}(I, n, n)
        # T_imp! = (du,u,p,t) → du = -u  ⟹  J = -I  ⟹  Wfact = dtγ*J - I = -dtγ*I - I
        prob = ODEProblem(
            ClimaODEFunction(;
                T_exp! = (du, u, p, t) -> (du .= 0.1 .* u),
                T_imp! = ODEFunction(
                    (du, u, p, t) -> (du .= -u);
                    jac_prototype = zeros(n, n),
                    Wfact = (W, u, p, dtγ, t) -> (W .= -dtγ .* Id .- Id),
                ),
            ),
            [1.0, 2.0],
            (0.0, 0.5),
            nothing,
        )
        alg = CTS.IMEXAlgorithm(ARS343(), NewtonsMethod(; max_iters = 2))
        sol = solve(prob, alg; dt = 0.01, save_everystep = true)

        for (i, t) in enumerate(sol.t)
            @test sol(t) == sol.u[i]
        end
    end

    # The remaining testsets pin the nearest-neighbor contract using a
    # hand-built solution with known, evenly-spaced save times, so the expected
    # index is independent of any floating-point solver output.
    times = [0.0, 1.0, 2.0, 3.0, 4.0]
    states = [[10.0 * i] for i in 0:4]   # distinct, easy-to-identify states
    sol = ODESolution(times, states, nothing, nothing)

    @testset "sol(t) returns a saved state, never an interpolant" begin
        # Halfway-ish queries must return one of the two stored states exactly,
        # not a blended value. 25.0 (the linear interpolant at t=2.5) must never
        # appear.
        for t in (0.4, 1.6, 2.5, 3.9)
            @test sol(t) in states
        end
    end

    @testset "Nearest-neighbor selection between saved points" begin
        @test sol(0.4) == states[1]   # closer to t=0.0
        @test sol(0.6) == states[2]   # closer to t=1.0
        @test sol(2.4) == states[3]   # closer to t=2.0
        @test sol(2.6) == states[4]   # closer to t=3.0
    end

    @testset "Tie-breaking: argmin returns the earlier point" begin
        # At an exact midpoint both neighbors are equidistant; argmin returns the
        # first (lower-index) minimum.
        @test sol(0.5) == states[1]
        @test sol(2.5) == states[3]
    end

    @testset "Out-of-range queries clamp to the endpoints" begin
        @test sol(-100.0) == states[1]     # before the first saved time
        @test sol(100.0) == states[end]    # after the last saved time
    end

    @testset "Exact match at every saved time" begin
        for (i, t) in enumerate(times)
            @test sol(t) == states[i]
        end
    end

    @testset "empty solution errors cleanly" begin
        empty_sol = ODESolution(Float64[], Vector{Float64}[], nothing, nothing)
        @test_throws ErrorException empty_sol(0.5)
    end
end
