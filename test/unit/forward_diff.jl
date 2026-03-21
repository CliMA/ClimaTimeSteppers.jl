#=
Automatic differentiation tests using ForwardDiff.jl.

Verifies that dual numbers propagate correctly through all solver families
(explicit RK, IMEX ARK, Rosenbrock) when differentiating with respect to
initial conditions and parameters.
=#
using ClimaTimeSteppers, ForwardDiff, LinearAlgebra, Test
import ClimaTimeSteppers as CTS

@testset "ForwardDiff compatibility" begin
    # Reference: du/dt = -λu, u(0) = u0  →  u(T) = u0 * exp(-λT)
    # At T=1, λ=1: u = u0 * exp(-1), du/du0 = exp(-1), du/dλ = -u0 * exp(-1)

    @testset "Explicit RK4 — d/du0" begin
        function solve_rk4_u0(u0_val)
            f = ClimaODEFunction(; T_exp! = (du, u, p, t) -> (du .= -u))
            prob = CTS.ODEProblem(f, [u0_val], (0.0, 1.0), nothing)
            sol = CTS.solve(prob, ExplicitAlgorithm(RK4()); dt = 0.1, saveat = (1.0,))
            return sol.u[end][1]
        end
        @test solve_rk4_u0(1.0) ≈ exp(-1.0) atol = 1e-4
        @test ForwardDiff.derivative(solve_rk4_u0, 1.0) ≈ exp(-1.0) atol = 1e-4
    end

    @testset "Explicit RK4 — ∇u0 (vector)" begin
        function solve_rk4_vec(u0)
            f = ClimaODEFunction(; T_exp! = (du, u, p, t) -> (du .= -u))
            prob = CTS.ODEProblem(f, copy(u0), (0.0, 1.0), nothing)
            sol = CTS.solve(prob, ExplicitAlgorithm(RK4()); dt = 0.1, saveat = (1.0,))
            return sum(sol.u[end])
        end
        grad = ForwardDiff.gradient(solve_rk4_vec, [1.0, 2.0])
        @test grad ≈ [exp(-1.0), exp(-1.0)] atol = 1e-4
    end

    @testset "Explicit RK4 — d/dλ (parameter)" begin
        function solve_rk4_param(λ)
            f = ClimaODEFunction(; T_exp! = (du, u, p, t) -> (du .= -p[1] .* u))
            prob = CTS.ODEProblem(f, [one(λ)], (zero(λ), one(λ)), [λ])
            sol = CTS.solve(
                prob, ExplicitAlgorithm(RK4()); dt = oftype(λ, 0.1), saveat = (one(λ),),
            )
            return sol.u[end][1]
        end
        @test solve_rk4_param(1.0) ≈ exp(-1.0) atol = 1e-4
        @test ForwardDiff.derivative(solve_rk4_param, 1.0) ≈ -exp(-1.0) atol = 1e-3
    end

    @testset "IMEX ARS343 — d/du0" begin
        function solve_imex_u0(u0_val)
            T = typeof(u0_val)
            function Wfact!(W, u, p, dtγ, t)
                fill!(W, zero(eltype(W)))
                for i in axes(W, 1)

                    W[i, i] = -dtγ - 1
                end
            end
            T_imp = CTS.ODEFunction(
                (du, u, p, t) -> (du .= -u);
                jac_prototype = zeros(T, 1, 1),
                Wfact = Wfact!,
            )
            f = ClimaODEFunction(;
                T_exp! = (du, u, p, t) -> (du .= zero(u)),
                T_imp! = T_imp,
            )
            prob = CTS.ODEProblem(f, [u0_val], (0.0, 1.0), nothing)
            alg = IMEXAlgorithm(
                ARS343(),
                NewtonsMethod(; max_iters = 1, update_j = UpdateEvery(NewTimeStep)),
            )
            sol = CTS.solve(prob, alg; dt = 0.1, saveat = (1.0,))
            return sol.u[end][1]
        end
        @test solve_imex_u0(1.0) ≈ exp(-1.0) atol = 1e-3
        @test ForwardDiff.derivative(solve_imex_u0, 1.0) ≈ exp(-1.0) atol = 1e-3
    end

    @testset "IMEX ARS343 — d/dλ (parameter)" begin
        function solve_imex_param(λ)
            T = typeof(λ)
            function Wfact!(W, u, p, dtγ, t)
                fill!(W, zero(eltype(W)))
                for i in axes(W, 1)

                    W[i, i] = -p[1] * dtγ - 1
                end
            end
            T_imp = CTS.ODEFunction(
                (du, u, p, t) -> (du .= -p[1] .* u);
                jac_prototype = zeros(T, 1, 1),
                Wfact = Wfact!,
            )
            f = ClimaODEFunction(;
                T_exp! = (du, u, p, t) -> (du .= zero(u)),
                T_imp! = T_imp,
            )
            prob = CTS.ODEProblem(f, [one(T)], (zero(T), one(T)), [λ])
            alg = IMEXAlgorithm(
                ARS343(),
                NewtonsMethod(; max_iters = 1, update_j = UpdateEvery(NewTimeStep)),
            )
            sol = CTS.solve(prob, alg; dt = T(0.1), saveat = (one(T),))
            return sol.u[end][1]
        end
        @test solve_imex_param(1.0) ≈ exp(-1.0) atol = 1e-3
        @test ForwardDiff.derivative(solve_imex_param, 1.0) ≈ -exp(-1.0) atol = 1e-2
    end

    @testset "Rosenbrock SSPKnoth — d/du0" begin
        function solve_rosenbrock_u0(u0_val)
            T = typeof(u0_val)
            function Wfact!(W, u, p, dtγ, t)
                fill!(W, zero(eltype(W)))
                for i in axes(W, 1)

                    W[i, i] = -dtγ - 1
                end
            end
            T_imp = CTS.ODEFunction(
                (du, u, p, t) -> (du .= -u);
                jac_prototype = zeros(T, 1, 1),
                Wfact = Wfact!,
            )
            f = ClimaODEFunction(; T_imp! = T_imp)
            prob = CTS.ODEProblem(f, [u0_val], (0.0, 1.0), nothing)
            alg = CTS.RosenbrockAlgorithm(CTS.tableau(SSPKnoth()))
            sol = CTS.solve(prob, alg; dt = 0.1, saveat = (1.0,))
            return sol.u[end][1]
        end
        @test solve_rosenbrock_u0(1.0) ≈ exp(-1.0) atol = 2e-3
        @test ForwardDiff.derivative(solve_rosenbrock_u0, 1.0) ≈ exp(-1.0) atol = 2e-3
    end

    @testset "Rosenbrock SSPKnoth — d/dλ (parameter)" begin
        function solve_rosenbrock_param(λ)
            T = typeof(λ)
            function Wfact!(W, u, p, dtγ, t)
                fill!(W, zero(eltype(W)))
                for i in axes(W, 1)

                    W[i, i] = -p[1] * dtγ - 1
                end
            end
            T_imp = CTS.ODEFunction(
                (du, u, p, t) -> (du .= -p[1] .* u);
                jac_prototype = zeros(T, 1, 1),
                Wfact = Wfact!,
            )
            f = ClimaODEFunction(; T_imp! = T_imp)
            prob = CTS.ODEProblem(f, [one(T)], (zero(T), one(T)), [λ])
            alg = CTS.RosenbrockAlgorithm(CTS.tableau(SSPKnoth()))
            sol = CTS.solve(prob, alg; dt = T(0.1), saveat = (one(T),))
            return sol.u[end][1]
        end
        @test solve_rosenbrock_param(1.0) ≈ exp(-1.0) atol = 2e-3
        @test ForwardDiff.derivative(solve_rosenbrock_param, 1.0) ≈ -exp(-1.0) atol = 2e-2
    end
end
