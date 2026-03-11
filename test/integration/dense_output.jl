#=
Dense output tests: verify that sol(t) interpolation is consistent.

CTS uses linear interpolation between saved steps for dense output.
=#
using ClimaTimeSteppers, DiffEqBase, LinearAlgebra, Test
import ClimaTimeSteppers as CTS

@testset "Dense output" begin

    hide_warning = (; kwargshandle = DiffEqBase.KeywordArgSilent)

    @testset "Explicit method: sol(t) matches at saved points" begin
        prob = ODEProblem(
            ClimaODEFunction(; T_exp! = (du, u, p, t) -> (du .= -u)),
            [1.0, 2.0],
            (0.0, 1.0),
            nothing,
        )
        alg = ExplicitAlgorithm(SSP33ShuOsher())
        sol = solve(prob, alg; dt = 0.01, save_everystep = true, hide_warning...)

        # At every saved timestep, sol(t) should exactly match the saved value
        for (i, t) in enumerate(sol.t)
            @test sol(t) ≈ sol.u[i] atol = 100 * eps(Float64)
        end
    end

    @testset "IMEX method: sol(t) matches at saved points" begin
        n = 2
        Id = Matrix{Float64}(I, n, n)
        # T_imp! = (du,u,p,t) → du = -u  ⟹  J = -I  ⟹  Wfact = dtγ*J - I = -dtγ*I - I
        prob = ODEProblem(
            ClimaODEFunction(;
                T_exp! = (du, u, p, t) -> (du .= 0.1 .* u),
                T_imp! = DiffEqBase.ODEFunction(
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
        sol = solve(prob, alg; dt = 0.01, save_everystep = true, hide_warning...)

        for (i, t) in enumerate(sol.t)
            @test sol(t) ≈ sol.u[i] atol = 100 * eps(Float64)
        end
    end

    @testset "Interpolation between saved points is bounded" begin
        # For du/dt = -u, u(t) = e^{-t}. The solution is monotonically decreasing.
        # CTS uses linear interpolation between saved steps, so the interpolated
        # value at a midpoint must lie strictly between the two saved endpoints.
        prob = ODEProblem(
            ClimaODEFunction(; T_exp! = (du, u, p, t) -> (du .= -u)),
            [1.0],
            (0.0, 1.0),
            nothing,
        )
        alg = ExplicitAlgorithm(RK4())
        sol = solve(prob, alg; dt = 0.1, save_everystep = true, hide_warning...)

        for i in 1:(length(sol.t) - 1)
            t_mid = (sol.t[i] + sol.t[i + 1]) / 2
            u_mid = sol(t_mid)
            u_lo = min(sol.u[i][1], sol.u[i + 1][1])
            u_hi = max(sol.u[i][1], sol.u[i + 1][1])
            # Linear interpolation must return a value between the two saved points
            @test u_lo ≤ u_mid[1] ≤ u_hi
        end
    end

    @testset "Interpolation accuracy via linear interpolation" begin
        # du/dt = -u, u(0) = 1 ⟹ u(t) = e^{-t}
        # CTS stores discrete values; we test linear interpolation accuracy
        # between saved points. Linear interpolation error is O(dt^2).
        prob = ODEProblem(
            ClimaODEFunction(; T_exp! = (du, u, p, t) -> (du .= -u)),
            [1.0],
            (0.0, 1.0),
            nothing,
        )
        alg = ExplicitAlgorithm(RK4())
        dt = 0.05
        sol = solve(prob, alg; dt, save_everystep = true, hide_warning...)

        # Manually linear-interpolate between consecutive saved points and check
        # against the analytic solution. Error bound: O(dt^2) for linear interp.
        for i in 1:(length(sol.t) - 1)
            t_mid = (sol.t[i] + sol.t[i + 1]) / 2
            # Linear interpolation between saved points
            u_lin = (sol.u[i][1] + sol.u[i + 1][1]) / 2
            u_exact = exp(-t_mid)
            @test abs(u_lin - u_exact) < dt^2
        end
    end
end
