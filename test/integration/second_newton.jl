#=
Test the second Newton solve feature (PR #392).

The second Newton code path runs T_imp_subproblem! first, then T_imp!.
Both implicit solves are applied in sequence, so a trivial (zero) subproblem
should give the same result as the standard single-solve path.
=#
using ClimaTimeSteppers, DiffEqBase, LinearAlgebra, Test
import ClimaTimeSteppers as CTS

@testset "Second Newton solve" begin
    n = 2
    λ₁ = 1.0
    λ₂ = 0.5
    hide_warning = (; kwargshandle = DiffEqBase.KeywordArgSilent)

    T_imp! = DiffEqBase.ODEFunction(
        (du, u, p, t) -> (du[1] = -λ₁ * u[1]; du[2] = -λ₂ * u[2] + u[1]);
        jac_prototype = zeros(n, n),
        Wfact = (W, u, p, dtγ, t) -> begin
            W[1, 1] = -λ₁ * dtγ - 1
            W[1, 2] = 0.0
            W[2, 1] = dtγ
            W[2, 2] = -λ₂ * dtγ - 1
        end,
    )

    @testset "Trivial subproblem matches standard path" begin
        # A zero subproblem is an identity solve (U' = U₀), so the full
        # solve's U₀ equals the original stage value, matching the standard path.
        T_imp_zero! = DiffEqBase.ODEFunction(
            (du, u, p, t) -> (du .= 0);
            jac_prototype = zeros(n, n),
            Wfact = (W, u, p, dtγ, t) -> (W .= -Matrix{Float64}(I, n, n)),
        )

        prob_sub = ODEProblem(
            ClimaODEFunction(;
                T_imp!,
                T_imp_subproblem! = T_imp_zero!,
                initialize_subproblem! = (u, p, γdt) -> nothing,
            ),
            copy([1.0, 1.0]),
            (0.0, 1.0),
            nothing,
        )
        alg_sub = CTS.IMEXAlgorithm(
            ARS232(),
            NewtonsMethod(; max_iters = 10),
            NewtonsMethod(; max_iters = 10),
        )
        sol_sub =
            solve(prob_sub, alg_sub; dt = 0.01, save_everystep = false, hide_warning...)

        prob_std =
            ODEProblem(ClimaODEFunction(; T_imp!), copy([1.0, 1.0]), (0.0, 1.0), nothing)
        alg_std = CTS.IMEXAlgorithm(ARS232(), NewtonsMethod(; max_iters = 10))
        sol_std =
            solve(prob_std, alg_std; dt = 0.01, save_everystep = false, hide_warning...)

        @test sol_sub.u[end] ≈ sol_std.u[end] rtol = 1e-10
    end

    @testset "Decoupled subproblem runs correctly" begin
        T_imp_sub! = DiffEqBase.ODEFunction(
            (du, u, p, t) -> (du[1] = -λ₁ * u[1]; du[2] = 0.0);
            jac_prototype = zeros(n, n),
            Wfact = (W, u, p, dtγ, t) -> begin
                W[1, 1] = -λ₁ * dtγ - 1
                W[1, 2] = 0.0
                W[2, 1] = 0.0
                W[2, 2] = -1.0
            end,
        )

        prob = ODEProblem(
            ClimaODEFunction(;
                T_imp!,
                T_imp_subproblem! = T_imp_sub!,
                initialize_subproblem! = (u, p, γdt) -> nothing,
            ),
            copy([1.0, 1.0]),
            (0.0, 1.0),
            nothing,
        )
        alg = CTS.IMEXAlgorithm(
            ARS232(),
            NewtonsMethod(; max_iters = 10),
            NewtonsMethod(; max_iters = 10),
        )
        sol = solve(prob, alg; dt = 0.01, save_everystep = false, hide_warning...)

        @test all(isfinite, sol.u[end])
        @test all(sol.u[end] .> 0)
    end
end
