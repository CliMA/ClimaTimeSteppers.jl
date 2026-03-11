# ============================================================================ #
# ARKode reference test cases (from "Example Programs for ARKode v4.4.0")
# ============================================================================ #

"""
Analytic scalar test problem from ARKode v4.4.0.
ODE: y' = λ*y + 1/(1+t^2) - λ*atan(t)
Linear implicitly-handled stiff term, explicitly-handled non-stiff source.
Analytic solution: y(t) = atan(t)
"""
function ark_analytic_test_cts(::Type{FT}) where {FT}
    λ = FT(-100)
    source(t) = 1 / (1 + t^2) - λ * atan(t)
    ClimaIntegratorTestCase(;
        test_name = "ark_analytic",
        linear_implicit = true,
        t_end = FT(10),
        Y₀ = FT[0],
        analytic_sol = (t) -> [atan(t)],
        tendency! = (Yₜ, Y, _, t) -> Yₜ .= λ .* Y .+ source(t),
        implicit_tendency! = (Yₜ, Y, _, t) -> Yₜ .= λ .* Y,
        explicit_tendency! = (Yₜ, Y, _, t) -> Yₜ .= source(t),
        Wfact! = (W, Y, _, dtγ, t) -> W .= dtγ * λ - 1,
        tgrad! = (∂Y∂t, Y, _, t) -> ∂Y∂t .= -(λ + 2 * t + λ * t^2) / (1 + t^2)^2,
        default_num_steps = 25000,
    )
end

"""
Nonlinear scalar test problem from ARKode v4.4.0.
ODE: y' = (t+1)*exp(-y)
Tests solvers on a fully nonlinear ODE without a stiff linear component.
Analytic solution: y(t) = log(t^2/2 + t + 1)
"""
function ark_analytic_nonlin_test_cts(::Type{FT}) where {FT}
    ClimaIntegratorTestCase(;
        test_name = "ark_analytic_nonlin",
        linear_implicit = false,
        t_end = FT(10),
        Y₀ = FT[0],
        analytic_sol = (t) -> [log(t^2 / 2 + t + 1)],
        tendency! = (Yₜ, Y, _, t) -> Yₜ .= (t + 1) .* exp.(.-Y),
        Wfact! = (W, Y, _, dtγ, t) -> W .= (-dtγ * (t + 1) .* exp.(.-Y) .- 1),
        tgrad! = (∂Y∂t, Y, _, t) -> ∂Y∂t .= exp.(.-Y),
        default_num_steps = 450,
    )
end

"""
Linear system test problem from ARKode v4.4.0.
ODE: Y' = A*Y
Tests a linearly implicitly-handled 3x3 coupled system with known eigenvalue spectrum.
"""
function ark_analytic_sys_test_cts(::Type{FT}) where {FT}
    λ = FT(-100)
    V = FT[1 -1 1; -1 2 1; 0 -1 2]
    V⁻¹ = FT[5 1 -3; 2 2 -2; 1 1 1] / 4
    D = Diagonal(FT[-1 / 2, -1 / 10, λ])
    A = V * D * V⁻¹
    I = LinearAlgebra.I(3)
    Y₀ = FT[1, 1, 1]
    ClimaIntegratorTestCase(;
        test_name = "ark_analytic_sys",
        linear_implicit = true,
        t_end = FT(1 / 20),
        Y₀,
        analytic_sol = (t) -> V * exp(D * t) * V⁻¹ * Y₀,
        tendency! = (Yₜ, Y, _, t) -> mul!(Yₜ, A, Y),
        Wfact! = (W, Y, _, dtγ, t) -> W .= dtγ .* A .- I,
        default_num_steps = 200,
    )
end

"""
One-way coupled multi-rate test problem from ARKode v4.4.0.
ODE: Y' = L*Y
Tests multi-rate methods with fast oscillatory modes and a slow decaying mode.
"""
function onewaycouple_mri_test_cts(::Type{FT}) where {FT}
    Y₀ = FT[1, 0, 2]
    function analytic_sol(t)
        Y = similar(Y₀)
        Y[1] = cos(50 * t)
        Y[2] = sin(50 * t)
        Y[3] = 5051 / 2501 * exp(-t) - 49 / 2501 * cos(50 * t) + 51 / 2501 * sin(50 * t)
        return Y
    end
    L = FT[0 -50 0; 50 0 0; 1 1 -1]
    I = LinearAlgebra.I(3)
    ClimaIntegratorTestCase(;
        test_name = "ark_onewaycouple_mri",
        linear_implicit = true,
        t_end = FT(1),
        Y₀ = FT[1, 0, 2],
        analytic_sol,
        tendency! = (Yₜ, Y, _, t) -> mul!(Yₜ, L, Y),
        Wfact! = (W, Y, _, dtγ, t) -> W .= dtγ .* L .- I,
        default_num_steps = 9000,
        high_order_sample_shifts = 3,
    )
end
