#=
Long-time stability tests: verify methods remain stable and accurate
over many (10⁴+) timesteps.
=#
using ClimaTimeSteppers, LinearAlgebra, Test
import ClimaTimeSteppers as CTS
import ClimaTimeSteppers: ODEProblem, ODEFunction, solve

@testset "Long-time stability" begin



    @testset "Explicit decay over 10⁴ steps" begin
        # du/dt = -u, u(0) = 1 ⟹ u(t) = exp(-t)
        # 10000 steps of dt=0.001 ⟹ t_end = 10, u_exact = exp(-10) ≈ 4.5e-5
        t_end = 10.0
        dt = 0.001
        prob = ODEProblem(
            ClimaODEFunction(; T_exp! = (du, u, p, t) -> (du .= -u)),
            [1.0],
            (0.0, t_end),
            nothing,
        )
        alg = ExplicitAlgorithm(RK4())
        sol = solve(prob, alg; dt, save_everystep = false)

        u_exact = exp(-t_end)
        @test sol.u[end][1] ≈ u_exact rtol = 1e-6
        @test isfinite(sol.u[end][1])
    end

    @testset "Explicit oscillator over 10⁴ steps preserves amplitude" begin
        # du₁/dt = u₂, du₂/dt = -u₁  ⟹ rotation, |u| should be conserved.
        # 10000 steps of dt=0.001 ⟹ t_end = 10, ~1.6 full orbits
        t_end = 10.0
        dt = 0.001
        prob = ODEProblem(
            ClimaODEFunction(;
                T_exp! = (du, u, p, t) -> (du[1] = u[2]; du[2] = -u[1]; nothing),
            ),
            [1.0, 0.0],
            (0.0, t_end),
            nothing,
        )
        alg = ExplicitAlgorithm(RK4())
        sol = solve(prob, alg; dt, save_everystep = false)

        initial_norm = norm([1.0, 0.0])
        final_norm = norm(sol.u[end])
        # RK4 is not symplectic, but amplitude drift should be tiny over 10⁴ steps
        @test abs(final_norm - initial_norm) / initial_norm < 1e-6
        @test sol.u[end][1] ≈ cos(t_end) atol = 1e-5
        @test sol.u[end][2] ≈ -sin(t_end) atol = 1e-5
    end

    @testset "IMEX decay over 10⁴ steps" begin
        # Same exponential decay but solved implicitly.
        n = 1
        t_end = 10.0
        dt = 0.001
        prob = ODEProblem(
            ClimaODEFunction(;
                T_imp! = ODEFunction(
                    (du, u, p, t) -> (du .= -u);
                    jac_prototype = zeros(n, n),
                    Wfact = (W, u, p, dtγ, t) -> (W[1, 1] = -dtγ - 1),
                ),
            ),
            [1.0],
            (0.0, t_end),
            nothing,
        )
        alg = CTS.IMEXAlgorithm(ARS232(), NewtonsMethod(; max_iters = 2))
        sol = solve(prob, alg; dt, save_everystep = false)

        u_exact = exp(-t_end)
        # Global error of 2nd-order method: O(t_end * dt²) = O(10 * dt²)
        @test abs(sol.u[end][1] - u_exact) < 10 * dt^2
        @test isfinite(sol.u[end][1])
    end

    @testset "Rosenbrock decay over 10⁴ steps" begin
        n = 1
        t_end = 10.0
        dt = 0.001
        Id = Matrix{Float64}(I, n, n)
        prob = ODEProblem(
            ClimaODEFunction(;
                T_imp! = ODEFunction(
                    (du, u, p, t) -> (du .= -u);
                    jac_prototype = zeros(n, n),
                    Wfact = (W, u, p, dtγ, t) -> (W[1, 1] = -dtγ - 1),
                ),
                T_exp! = (du, u, p, t) -> (du .= 0),
            ),
            [1.0],
            (0.0, t_end),
            nothing,
        )
        alg = CTS.RosenbrockAlgorithm(CTS.tableau(CTS.SSPKnoth()))
        sol = solve(prob, alg; dt, save_everystep = false)

        u_exact = exp(-t_end)
        @test abs(sol.u[end][1] - u_exact) < 10 * dt^2
        @test isfinite(sol.u[end][1])
    end
end
