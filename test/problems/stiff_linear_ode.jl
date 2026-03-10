# ============================================================================ #
# Stiff IMEX test problem
# ============================================================================ #

"""
Stiff linear IMEX problem with known eigenvalue separation.

System: du/dt = A_stiff * u + A_nonstiff * u
where A_stiff has eigenvalue -λ (stiff, implicit) and A_nonstiff has
eigenvalue -1 (non-stiff, explicit).

Specifically:
  du₁/dt = -λ * u₁ + u₂
  du₂/dt = u₁ - u₂

with λ = 1000 (stiff). Solution is u(t) = exp(A*t) * u₀.
"""
function stiff_linear_test_cts(::Type{FT}) where {FT}
    λ = FT(1000)
    A_imp = FT[-λ 0; 0 0]  # stiff part (diagonal)
    A_exp = FT[0 1; 1 -1]  # non-stiff part
    A = A_imp + A_exp
    Id = LinearAlgebra.I(2)
    Y₀ = FT[1, 1]
    ClimaIntegratorTestCase(;
        test_name = "stiff_linear",
        linear_implicit = true,
        t_end = FT(0.01),
        Y₀,
        analytic_sol = (t) -> exp(A * t) * Y₀,
        tendency! = (Yₜ, Y, _, t) -> mul!(Yₜ, A, Y),
        implicit_tendency! = (Yₜ, Y, _, t) -> mul!(Yₜ, A_imp, Y),
        explicit_tendency! = (Yₜ, Y, _, t) -> mul!(Yₜ, A_exp, Y),
        Wfact! = (W, Y, _, Δt, t) -> W .= Δt .* A_imp .- Id,
        default_num_steps = 1000,
    )
end
