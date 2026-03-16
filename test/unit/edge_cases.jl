#=
Edge case and error handling tests.
=#
using ClimaTimeSteppers, DiffEqBase, LinearAlgebra, Test
import ClimaTimeSteppers as CTS

@testset "Edge cases" begin

    hide_warning = (; kwargshandle = DiffEqBase.KeywordArgSilent)

    @testset "NaN initial conditions propagate" begin
        prob = ODEProblem(
            ClimaODEFunction(; T_exp! = (du, u, p, t) -> (du .= -u)),
            [NaN],
            (0.0, 0.1),
            nothing,
        )
        alg = ExplicitAlgorithm(SSP33ShuOsher())
        sol = solve(prob, alg; dt = 0.01, save_everystep = false, hide_warning...)
        @test all(isnan, sol.u[end])
    end

    @testset "Inf initial conditions propagate" begin
        prob = ODEProblem(
            ClimaODEFunction(; T_exp! = (du, u, p, t) -> (du .= -u)),
            [Inf],
            (0.0, 0.1),
            nothing,
        )
        alg = ExplicitAlgorithm(SSP33ShuOsher())
        sol = solve(prob, alg; dt = 0.01, save_everystep = false, hide_warning...)
        @test all(isnan, sol.u[end]) || all(isinf, sol.u[end])
    end

    @testset "Zero initial conditions stay zero (linear ODE)" begin
        prob = ODEProblem(
            ClimaODEFunction(; T_exp! = (du, u, p, t) -> (du .= -u)),
            [0.0, 0.0],
            (0.0, 1.0),
            nothing,
        )
        alg = ExplicitAlgorithm(RK4())
        sol = solve(prob, alg; dt = 0.01, save_everystep = false, hide_warning...)
        @test sol.u[end] == [0.0, 0.0]
    end

    @testset "Very small dt produces accurate result" begin
        prob = ODEProblem(
            ClimaODEFunction(; T_exp! = (du, u, p, t) -> (du .= -u)),
            [1.0],
            (0.0, 0.01),
            nothing,
        )
        alg = ExplicitAlgorithm(SSP22Heuns())
        sol = solve(prob, alg; dt = 1e-6, save_everystep = false, hide_warning...)
        @test sol.u[end][1] ≈ exp(-0.01) rtol = 1e-8
    end

    @testset "Single step integration" begin
        prob = ODEProblem(
            ClimaODEFunction(; T_exp! = (du, u, p, t) -> (du .= -u)),
            [1.0],
            (0.0, 0.01),
            nothing,
        )
        alg = ExplicitAlgorithm(SSP33ShuOsher())
        sol = solve(prob, alg; dt = 0.01, save_everystep = false, hide_warning...)
        @test length(sol.u) == 2  # initial + final
        @test sol.u[end][1] < 1.0  # decaying
    end

    @testset "IMEX with singular-like Jacobian (zero implicit tendency)" begin
        # T_imp! = 0 means Wfact = -I, which is well-conditioned.
        # This tests that zero implicit tendency doesn't cause issues.
        n = 2
        Id = Matrix{Float64}(I, n, n)
        prob = ODEProblem(
            ClimaODEFunction(;
                T_exp! = (du, u, p, t) -> (du .= -u),
                T_imp! = DiffEqBase.ODEFunction(
                    (du, u, p, t) -> (du .= 0);
                    jac_prototype = zeros(n, n),
                    Wfact = (W, u, p, γ, t) -> (W .= -Id),
                ),
            ),
            [1.0, 2.0],
            (0.0, 0.5),
            nothing,
        )
        alg = CTS.IMEXAlgorithm(ARS232(), NewtonsMethod(; max_iters = 2))
        sol = solve(prob, alg; dt = 0.01, save_everystep = false, hide_warning...)
        @test sol.u[end][1] ≈ exp(-0.5) atol = 0.01
        @test sol.u[end][2] ≈ 2 * exp(-0.5) atol = 0.02
    end

    @testset "Newton's method with zero initial residual" begin
        # If x_init is already the solution, Newton should not diverge
        f!(f, x) = f .= x .- [1.0, 2.0]
        j!(j, x) = j .= Matrix{Float64}(I, 2, 2)
        x = [1.0, 2.0]  # already exact
        alg = NewtonsMethod(; max_iters = 5)
        cache = CTS.allocate_cache(alg, x, zeros(2, 2))
        CTS.solve_newton!(alg, cache, x, f!, j!)
        @test x ≈ [1.0, 2.0]
    end
end
