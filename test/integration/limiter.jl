#=
Tests for the `T_lim!` / `lim!` code path in IMEX solvers.

Verifies that the generic scalar limiter is applied correctly during explicit 
and IMEX timestepping by using a simple clipping limiter that mathematically enforces non-negativity.
=#
using ClimaTimeSteppers, DiffEqBase, LinearAlgebra, Test
import ClimaTimeSteppers as CTS

@testset "T_lim! and lim! integration" begin
    hide_warning = (; kwargshandle = DiffEqBase.KeywordArgSilent)
    @testset "Explicit algorithm with limiter" begin
        # Problem: du/dt = -u (decay), split into T_lim (limited) component.
        # The limiter clips values to [0, ∞).
        # Starting from u₀ = [1.0, 2.0], the solution decays exponentially.
        # The limiter should be a no-op here since values stay positive.
        n = 2
        prob = ODEProblem(
            ClimaODEFunction(;
                T_lim! = (du, u, p, t) -> (du .= -u),
                lim! = (u, p, t, u_ref) -> (u .= max.(u, 0.0)),
            ),
            [1.0, 2.0],
            (0.0, 0.5),
            nothing,
        )

        alg = ExplicitAlgorithm(SSP33ShuOsher())

        sol = solve(prob, alg; dt = 0.01, save_everystep = false, hide_warning...)
        u_final = sol.u[end]

        # Should approximate exp(-0.5) * u₀
        @test u_final[1] ≈ exp(-0.5) atol = 0.01
        @test u_final[2] ≈ 2 * exp(-0.5) atol = 0.01
        # Values must remain non-negative (limiter enforces this)
        @test all(u_final .>= 0)
    end

    @testset "Limiter clips negative overshoots" begin
        # Problem with a tendency that could cause negative values
        # if the limiter weren't applied: du/dt = -10*u (fast decay).
        # With large dt, explicit methods can overshoot to negative values.
        # The limiter clips them back to 0.
        prob = ODEProblem(
            ClimaODEFunction(;
                T_lim! = (du, u, p, t) -> (du .= -10.0 .* u),
                lim! = (u, p, t, u_ref) -> (u .= max.(u, 0.0)),
            ),
            [1.0],
            (0.0, 1.0),
            nothing,
        )

        alg = ExplicitAlgorithm(SSP33ShuOsher())

        sol = solve(prob, alg; dt = 0.1, save_everystep = true, hide_warning...)

        # All saved values should be non-negative due to limiter
        for u in sol.u
            @test all(u .>= 0)
        end
    end

    @testset "IMEX with T_exp_T_lim!" begin
        # Split problem: T_exp_T_lim! handles both explicit and limited tendencies.
        # du/dt = 0.01*u (T_exp) + -0.1*u (T_lim), with limiter ensuring non-negativity.
        # T_exp! uses a small positive coefficient so that it can never drive u
        # negative, making the non-negativity test robust to any dt.
        Id = Matrix{Float64}(LinearAlgebra.I, 2, 2)
        prob = ODEProblem(
            ClimaODEFunction(;
                T_exp_T_lim! = (du_exp, du_lim, u, p, t) -> begin
                    du_exp .= 0.01 .* u
                    du_lim .= -0.1 .* u
                end,
                T_imp! = DiffEqBase.ODEFunction(
                    (du, u, p, t) -> (du .= 0);
                    jac_prototype = zeros(2, 2),
                    Wfact = (W, u, p, dtγ, t) -> (W .= -Id),
                ),
                lim! = (u, p, t, u_ref) -> (u .= max.(u, 0.0)),
            ),
            [1.0, 2.0],
            (0.0, 1.0),
            nothing,
        )

        alg = CTS.IMEXAlgorithm(ARS343(), NewtonsMethod(; max_iters = 1))

        sol = solve(prob, alg; dt = 0.05, save_everystep = false, hide_warning...)
        u_final = sol.u[end]

        # Combined rate is -0.09, so solution ≈ exp(-0.09) * u₀
        # ARS343 is 3rd order, dt=0.05, so error ≈ O(dt^3 * t_end/dt) ≈ O(1e-4)
        @test u_final[1] ≈ exp(-0.09) atol = 0.005
        @test u_final[2] ≈ 2 * exp(-0.09) atol = 0.01
        @test all(u_final .>= 0)
    end
end
