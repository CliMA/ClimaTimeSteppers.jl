#=
SSP monotonicity preservation tests.

SSP (Strong Stability Preserving) methods guarantee that if forward Euler
preserves a convex property (e.g., non-negativity) under a CFL constraint
dt ≤ dt_FE, then the SSP method preserves it under dt ≤ c * dt_FE, where
c is the SSP coefficient.

We test this by using a problem where forward Euler preserves non-negativity
for dt ≤ 1/λ, and verify that SSP methods also preserve it.
=#
using ClimaTimeSteppers, Test
import ClimaTimeSteppers: ODEProblem, solve

@testset "SSP monotonicity" begin



    @testset "Explicit SSP methods preserve non-negativity" begin
        # du/dt = -λ * u, forward Euler: u_{n+1} = (1 - λ*dt) * u_n
        # Non-negative for dt ≤ 1/λ. With λ=1, dt_FE = 1.
        # SSP22Heuns has SSP coefficient c=1, so dt ≤ 1 preserves non-negativity.
        # SSP33ShuOsher has SSP coefficient c=1, so dt ≤ 1.
        λ = 1.0
        dt_safe = 0.9  # well within CFL bound

        for (name, alg) in [
            ("SSP22Heuns", ExplicitAlgorithm(SSP22Heuns())),
            ("SSP33ShuOsher", ExplicitAlgorithm(SSP33ShuOsher())),
        ]
            @testset "$name" begin
                prob = ODEProblem(
                    ClimaODEFunction(; T_exp! = (du, u, p, t) -> (du .= -λ .* u)),
                    [1.0, 0.5, 0.1],  # all positive
                    (0.0, 10.0),
                    nothing,
                )
                sol = solve(prob, alg; dt = dt_safe, save_everystep = true)
                # All states at all times must be non-negative
                for u in sol.u
                    @test all(u .>= 0)
                end
            end
        end
    end

    @testset "Non-SSP RK4 amplifies outside stability region" begin
        # RK4 is not SSP. Outside its stability region (λ*dt > 2.78),
        # the solution blows up instead of decaying.
        λ = 10.0
        prob = ODEProblem(
            ClimaODEFunction(; T_exp! = (du, u, p, t) -> (du .= -λ .* u)),
            [1.0],
            (0.0, 2.0),
            nothing,
        )
        alg = ExplicitAlgorithm(RK4())
        sol = solve(prob, alg; dt = 0.5, save_everystep = false)
        # With λ*dt = 5, RK4 is outside its stability region.
        # The solution should blow up rather than decay to ~exp(-20) ≈ 2e-9.
        @test abs(sol.u[end][1]) > 1.0
    end

    @testset "SSP method with limiter enforces bounds" begin
        # Even with a tendency that could cause overshoots, the SSP method
        # combined with a limiter maintains the invariant.
        λ = 5.0
        prob = ODEProblem(
            ClimaODEFunction(;
                T_lim! = (du, u, p, t) -> (du .= -λ .* u),
                lim! = (u, p, t, u_ref) -> (u .= max.(u, 0.0)),
            ),
            [1.0, 0.5],
            (0.0, 5.0),
            nothing,
        )
        alg = ExplicitAlgorithm(SSP33ShuOsher())
        sol = solve(prob, alg; dt = 0.5, save_everystep = true)
        for u in sol.u
            @test all(u .>= 0)
        end
    end
end
