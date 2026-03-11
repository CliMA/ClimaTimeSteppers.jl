#=
Jacobian accuracy tests: verify that user-provided Wfact matches
a finite-difference approximation of the true Jacobian of T_imp!.
=#
using ClimaTimeSteppers, DiffEqBase, LinearAlgebra, Test
import ClimaTimeSteppers as CTS

@testset "Jacobian accuracy" begin

    @testset "Linear T_imp! Jacobian matches Wfact" begin
        # T_imp!(du, u) = A * u ⟹ J = A, Wfact = dtγ * A - I
        n = 3
        A = [-2.0 0.5 0.0; 0.3 -1.0 0.1; 0.0 0.2 -3.0]
        Id = Matrix{Float64}(I, n, n)

        T_imp! = DiffEqBase.ODEFunction(
            (du, u, p, t) -> mul!(du, A, u);
            jac_prototype = zeros(n, n),
            Wfact = (W, u, p, dtγ, t) -> (W .= dtγ .* A .- Id),
        )

        # Finite-difference Jacobian at a test point
        u0 = [1.0, 2.0, 3.0]
        dtγ = 0.1
        t = 0.0
        ε = 1e-7

        J_fd = zeros(n, n)
        du0 = zeros(n)
        du_pert = zeros(n)
        T_imp!.f(du0, u0, nothing, t)
        for j in 1:n
            u_pert = copy(u0)
            u_pert[j] += ε
            T_imp!.f(du_pert, u_pert, nothing, t)
            J_fd[:, j] = (du_pert .- du0) ./ ε
        end

        # Wfact should equal dtγ * J - I
        W_expected = dtγ .* J_fd .- Id
        W_actual = zeros(n, n)
        T_imp!.Wfact(W_actual, u0, nothing, dtγ, t)

        @test W_actual ≈ W_expected atol = 1e-6
    end

    @testset "Nonlinear T_imp! Jacobian matches Wfact" begin
        # T_imp!(du, u) = [-u₁²; -u₁*u₂] ⟹ J = [-2u₁ 0; -u₂ -u₁]
        n = 2
        Id = Matrix{Float64}(I, n, n)

        T_imp! = DiffEqBase.ODEFunction(
            (du, u, p, t) -> (du[1] = -u[1]^2; du[2] = -u[1] * u[2]);
            jac_prototype = zeros(n, n),
            Wfact = (W, u, p, dtγ, t) -> begin
                W[1, 1] = dtγ * (-2 * u[1]) - 1
                W[1, 2] = 0.0
                W[2, 1] = dtγ * (-u[2])
                W[2, 2] = dtγ * (-u[1]) - 1
            end,
        )

        u0 = [1.5, 2.5]
        dtγ = 0.05
        t = 0.0
        ε = 1e-7

        J_fd = zeros(n, n)
        du0 = zeros(n)
        du_pert = zeros(n)
        T_imp!.f(du0, u0, nothing, t)
        for j in 1:n
            u_pert = copy(u0)
            u_pert[j] += ε
            T_imp!.f(du_pert, u_pert, nothing, t)
            J_fd[:, j] = (du_pert .- du0) ./ ε
        end

        W_expected = dtγ .* J_fd .- Id
        W_actual = zeros(n, n)
        T_imp!.Wfact(W_actual, u0, nothing, dtγ, t)

        @test W_actual ≈ W_expected atol = 1e-5
    end

    @testset "Incorrect Wfact is detected" begin
        # Deliberately wrong Wfact (missing a factor) to ensure the test
        # methodology can actually detect a bug.
        n = 2
        A = [-1.0 0.5; 0.3 -2.0]
        Id = Matrix{Float64}(I, n, n)

        T_imp! = DiffEqBase.ODEFunction(
            (du, u, p, t) -> mul!(du, A, u);
            jac_prototype = zeros(n, n),
            Wfact = (W, u, p, dtγ, t) -> (W .= dtγ .* (2 .* A) .- Id),  # 2× wrong!
        )

        u0 = [1.0, 1.0]
        dtγ = 0.1
        t = 0.0
        ε = 1e-7

        J_fd = zeros(n, n)
        du0 = zeros(n)
        du_pert = zeros(n)
        T_imp!.f(du0, u0, nothing, t)
        for j in 1:n
            u_pert = copy(u0)
            u_pert[j] += ε
            T_imp!.f(du_pert, u_pert, nothing, t)
            J_fd[:, j] = (du_pert .- du0) ./ ε
        end

        W_expected = dtγ .* J_fd .- Id
        W_actual = zeros(n, n)
        T_imp!.Wfact(W_actual, u0, nothing, dtγ, t)

        # This should NOT match because Wfact is deliberately wrong
        @test !(W_actual ≈ W_expected)
    end
end
