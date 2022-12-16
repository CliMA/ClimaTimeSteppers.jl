using DiffEqBase, ClimaTimeSteppers, LinearAlgebra, StaticArrays

"""
Single variable linear ODE

```math
\\frac{du}{dt} = αu
```
with initial condition ``u_0=\\frac{1}{2}``, parameter ``α=1.01``, and solution
```math
u(t) = u_0 e^{αt}
```

This is an in-place variant of the one from DiffEqProblemLibrary.jl.
"""
function linear_prob()
    ODEProblem(
        IncrementingODEFunction{true}((du, u, p, t, α = true, β = false) -> (du .= α .* p .* u .+ β .* du)),
        [1 / 2],
        (0.0, 1.0),
        1.01,
    )
end
function linear_prob_fe()
    ODEProblem(ForwardEulerODEFunction((un, u, p, t, dt) -> (un .= u .+ dt .* p .* u)), [1.0], (0.0, 1.0), -0.2)
end

function linear_prob_wfactt()
    ODEProblem(
        ODEFunction(
            (du, u, p, t) -> (du .= p .* u);
            jac_prototype = zeros(1, 1),
            Wfact_t = (W, u, p, γ, t) -> (W[1, 1] = 1 / γ - p),
        ),
        [1 / 2],
        (0.0, 1.0),
        -0.2,
    )
end

function split_linear_prob_wfact_split()
    ODEProblem(
        ClimaODEFunction(;
            T_imp! = ODEFunction(
                (du, u, p, t) -> (du .= real(p) .* u);
                jac_prototype = zeros(ComplexF64, 1, 1),
                Wfact = (W, u, p, γ, t) -> (W[1, 1] = γ * real(p) - 1),
            ),
            T_exp! = ODEFunction((du, u, p, t) -> (du .= imag(p) * im .* u)),
        ),
        [1 / 2 + 0.0 * im],
        (0.0, 1.0),
        -0.2 + 0.1 * im,
    )
end

function split_linear_prob_wfact_split_fe()
    ODEProblem(
        ClimaODEFunction(;
            T_imp! = ODEFunction(
                (du, u, p, t) -> (du .= real(p) .* u);
                jac_prototype = zeros(ComplexF64, 1, 1),
                Wfact = (W, u, p, γ, t) -> (W[1, 1] = γ * real(p) - 1),
            ),
            T_exp! = (du, u, p, t) -> (du .= imag(p) * im .* u),
        ),
        [1 / 2 + 0.0 * im],
        (0.0, 1.0),
        -0.2 + 0.1 * im,
    )
end

# DiffEqProblemLibrary.jl uses the `analytic` argument to ODEFunction to store the exact solution
# IncrementingODEFunction doesn't have that
function linear_sol(u0, p, t)
    u0 .* exp(p * t)
end


"""
Two variable linear ODE

```math
\\frac{du_1}{dt} = α u_2 \\quad \\frac{du_2}{dt} = -α u_1
```
with initial condition ``u_0=[0,1]``, parameter ``α=2``, and solution
```math
u(t) = [cos(αt) sin(αt); -sin(αt) cos(αt) ] u_0
```
"""
function sincos_prob()
    ODEProblem(
        IncrementingODEFunction{true}((du, u, p, t, α = true, β = false) -> (du[1] = α * p * u[2] + β * du[1];
        du[2] = -α * p * u[1] + β * du[2])),
        [0.0, 1.0],
        (0.0, 1.0),
        2.0,
    )
end
function sincos_prob_fe()
    ODEProblem(
        ForwardEulerODEFunction((un, u, p, t, dt) -> (un[1] = u[1] + dt * p * u[2]; un[2] = u[2] - dt * p * u[1])),
        [0.0, 1.0],
        (0.0, 1.0),
        2.0,
    )
end

function sincos_sol(u0, p, t)
    s, c = sincos(p * t)
    [c s; -s c] * u0
end

"""
IMEX problem with autonomous linear part

```math
\\frac{du}{dt} = u + cos(t)/α
```
with initial condition ``u_0 = 1/2``, parameter ``α=4``, and solution
```math
u(t) = \frac{e^t + sin(t) - cos(t)}{2 α} + u_0 e^t
```
"""
function imex_autonomous_prob(::Type{ArrayType}) where {ArrayType}
    SplitODEProblem(
        IncrementingODEFunction{true}((du, u, p, t, α = true, β = false) -> (du .= α .* u .+ β .* du)),
        IncrementingODEFunction{true}((du, u, p, t, α = true, β = false) -> (du .= α .* cos(t) / p .+ β .* du)),
        ArrayType([0.5]),
        (0.0, 1.0),
        4.0,
    )
end

function linsolve_direct(::Type{Val{:init}}, f, u0; kwargs...)
    function _linsolve!(x, A, b, update_matrix = false; kwargs...)
        x .= A \ b
    end
end

function imex_autonomous_prob_jac(::Type{ArrayType}) where {ArrayType}
    ODEProblem(
        ODEFunction(
            IncrementingODEFunction{true}(
                (du, u, p, t, α = true, β = false) -> (du .= α .* (u .+ cos(t) / p) .+ β .* du),
            ),
            jac_prototype = zeros(1, 1),
            Wfact = (W, u, p, γ, t) -> W[1, 1] = 1 - γ,
        ),
        ArrayType([0.5]),
        (0.0, 1.0),
        4.0,
    )
end

function imex_autonomous_sol(u0, p, t)
    (exp(t) + sin(t) - cos(t)) / 2p .+ exp(t) .* u0
end

function imex_nonautonomous_prob(::Type{ArrayType}) where {ArrayType}
    SplitODEProblem(
        IncrementingODEFunction{true}((du, u, p, t, α = true, β = false) -> (du .= α .* cos(t) * u .+ β .* du)),
        IncrementingODEFunction{true}((du, u, p, t, α = true, β = false) -> (du .= α .* cos(t) / p .+ β .* du)),
        ArrayType([0.5]),
        (0.0, 2.0),
        4.0,
    )
end

function imex_nonautonomous_sol(u0, p, t)
    (exp(sin(t)) .* (1 .+ p .* u0) .- 1) ./ p
end

function kpr_rhs(Q, param, t)
    yf, ys = Q
    ω, λf, λs, ξ, α = param

    ηfs = ((1 - ξ) / α) * (λf - λs)
    ηsf = -ξ * α * (λf - λs)
    Ω = @SMatrix [
        λf ηfs
        ηsf λs
    ]

    g = @SVector [(-3 + yf^2 - cos(ω * t)) / 2yf, (-2 + ys^2 - cos(t)) / 2ys]
    h = @SVector [ω * sin(ω * t) / 2yf, sin(t) / 2ys]
    return Ω * g - h
end

function kpr_fast!(dQ, Q, param, t, α = 1, β = 0)
    dQf, _ = kpr_rhs(Q, param, t)
    dQ[1] = α * dQf + β * dQ[1]
    dQ[2] = β * dQ[2]
end

function kpr_slow!(dQ, Q, param, t, α = 1, β = 0)
    _, dQs = kpr_rhs(Q, param, t)
    dQ[1] = β * dQ[1]
    dQ[2] = α * dQs + β * dQ[2]
end

function kpr_sol(u0, param, t)
    #@assert u0 == [sqrt(4) sqrt(3)]
    ω, λf, λs, ξ, α = param
    return [
        sqrt(3 + cos(ω * t))
        sqrt(2 + cos(t))
    ]
end

"""
Test problem (4.2) from RobertsSarsharSandu2018arxiv

# TODO: move this to bibliography

```
@article{RobertsSarsharSandu2018arxiv,
    title={Coupled Multirate Infinitesimal GARK Schemes for Stiff Systems with
            Multiple Time Scales},
    author={Roberts, Steven and Sarshar, Arash and Sandu, Adrian},
    journal={arXiv preprint arXiv:1812.00808},
    year={2019}
}
```

Note: The actual rates are all over the place with this test and passing largely
        depends on final dt size
"""
function kpr_multirate_prob()
    kpr_param = (ω = 100, λf = -10, λs = -1, ξ = 0.1, α = 1)
    SplitODEProblem(
        IncrementingODEFunction{true}(kpr_fast!),
        IncrementingODEFunction{true}(kpr_slow!),
        [sqrt(4), sqrt(3)],
        (0.0, 5π / 2),
        kpr_param,
    )
end

function kpr_singlerate_prob()
    kpr_param = (ω = 100, λf = -10, λs = -1, ξ = 0.1, α = 1)
    ODEProblem(
        IncrementingODEFunction{true}((dQ, Q, param, t, α = 1, β = 0) -> (dQ .= α .* kpr_rhs(Q, param, t) .+ β .* dQ)),
        [sqrt(4), sqrt(3)],
        (0.0, 5π / 2),
        kpr_param,
    )
end

reverse_problem(prob, analytic_sol) = ODEProblem(prob.f, analytic_sol(prob.tspan[2]), reverse(prob.tspan), prob.p)

struct IntegratorTestCase{FT, A, P, IP, SP, SIP}
    test_name::String
    linear_implicit::Bool
    t_end::FT
    analytic_sol::A
    prob::P
    increment_prob::IP
    split_prob::SP
    split_increment_prob::SIP
end

function IntegratorTestCase(;
    test_name,
    linear_implicit,
    t_end,
    Y₀,
    analytic_sol,
    tendency!,
    increment!,
    implicit_tendency! = nothing,
    explicit_tendency! = nothing,
    implicit_increment! = nothing,
    explicit_increment! = nothing,
    Wfact!,
    tgrad! = nothing,
)
    FT = typeof(t_end)
    jac_prototype = Matrix{FT}(undef, length(Y₀), length(Y₀))
    func_args = (; jac_prototype, Wfact = Wfact!, tgrad = tgrad!)
    tendency_func = ODEFunction(tendency!; func_args...)
    increment_func = ForwardEulerODEFunction(increment!; func_args...)
    if isnothing(implicit_tendency!) # assume that related args are also nothing
        no_tendency!(Yₜ, Y, _, t) = Yₜ .= FT(0)
        no_increment!(Y⁺, Y, _, t, Δt) = Y⁺
        split_tendency_func = SplitFunction(tendency_func, no_tendency!)
        split_increment_func = SplitFunction(increment_func, ForwardEulerODEFunction(no_increment!))
    else
        split_tendency_func = SplitFunction(ODEFunction(implicit_tendency!; func_args...), explicit_tendency!)
        split_increment_func = SplitFunction(
            ForwardEulerODEFunction(implicit_increment!; func_args...),
            ForwardEulerODEFunction(explicit_increment!),
        )
    end
    make_prob(func) = ODEProblem(func, Y₀, (FT(0), t_end), nothing)
    IntegratorTestCase(
        test_name,
        linear_implicit,
        t_end,
        analytic_sol,
        make_prob(tendency_func),
        make_prob(increment_func),
        make_prob(split_tendency_func),
        make_prob(split_increment_func),
    )
end

function ClimaIntegratorTestCase(;
    test_name,
    linear_implicit,
    t_end,
    Y₀,
    analytic_sol,
    tendency!,
    increment!,
    implicit_tendency! = nothing,
    explicit_tendency! = nothing,
    implicit_increment! = nothing,
    explicit_increment! = nothing,
    Wfact!,
    tgrad! = nothing,
)
    FT = typeof(t_end)
    jac_prototype = Matrix{FT}(undef, length(Y₀), length(Y₀))
    func_args = (; jac_prototype, Wfact = Wfact!, tgrad = tgrad!)
    tendency_func = ClimaODEFunction(; T_imp! = ODEFunction(tendency!; func_args...))
    if isnothing(implicit_tendency!) # assume that related args are also nothing
        split_tendency_func = ClimaODEFunction(; T_imp! = ODEFunction(tendency!; func_args...))
    else
        split_tendency_func =
            ClimaODEFunction(; T_exp! = explicit_tendency!, T_imp! = ODEFunction(implicit_tendency!; func_args...))
    end
    make_prob(func) = ODEProblem(func, Y₀, (FT(0), t_end), nothing)
    IntegratorTestCase(
        test_name,
        linear_implicit,
        t_end,
        analytic_sol,
        make_prob(tendency_func),
        nothing,
        make_prob(split_tendency_func),
        nothing,
    )
end

# A trivial test case for which any value of dt will work
function constant_tendency_test(::Type{FT}) where {FT}
    tendency = FT[1, 2, 3]
    IntegratorTestCase(;
        test_name = "constant_tendency",
        linear_implicit = true,
        t_end = FT(1),
        Y₀ = FT[0, 0, 0],
        analytic_sol = (t) -> tendency .* t,
        tendency! = (Yₜ, Y, _, t) -> Yₜ .= tendency,
        increment! = (Y⁺, Y, _, t, Δt) -> Y⁺ .+= Δt .* tendency,
        Wfact! = (W, Y, _, Δt, t) -> W .= -1,
    )
end

function clima_constant_tendency_test(::Type{FT}) where {FT}
    tendency = FT[1, 2, 3]
    ClimaIntegratorTestCase(;
        test_name = "constant_tendency",
        linear_implicit = true,
        t_end = FT(1),
        Y₀ = FT[0, 0, 0],
        analytic_sol = (t) -> tendency .* t,
        tendency! = (Yₜ, Y, _, t) -> Yₜ .= tendency,
        increment! = (Y⁺, Y, _, t, Δt) -> Y⁺ .+= Δt .* tendency,
        Wfact! = (W, Y, _, Δt, t) -> W .= -1,
    )
end

# From Section 1.1 of "Example Programs for ARKode v4.4.0" by D. R. Reynolds
function ark_analytic_test_cts(::Type{FT}) where {FT}
    λ = FT(-100) # increase magnitude for more stiffness
    source(t) = 1 / (1 + t^2) - λ * atan(t)
    ClimaIntegratorTestCase(;
        test_name = "ark_analytic",
        linear_implicit = true,
        t_end = FT(10),
        Y₀ = FT[0],
        analytic_sol = (t) -> [atan(t)],
        tendency! = (Yₜ, Y, _, t) -> Yₜ .= λ .* Y .+ source(t),
        increment! = (Y⁺, Y, _, t, Δt) -> Y⁺ .+= Δt .* (λ .* Y .+ source(t)),
        implicit_tendency! = (Yₜ, Y, _, t) -> Yₜ .= λ .* Y,
        explicit_tendency! = (Yₜ, Y, _, t) -> Yₜ .= source(t),
        implicit_increment! = (Y⁺, Y, _, t, Δt) -> Y⁺ .+= (Δt * λ) .* Y,
        explicit_increment! = (Y⁺, Y, _, t, Δt) -> Y⁺ .+= Δt * source(t),
        Wfact! = (W, Y, _, Δt, t) -> W .= Δt * λ - 1,
        tgrad! = (∂Y∂t, Y, _, t) -> ∂Y∂t .= -(λ + 2 * t + λ * t^2) / (1 + t^2)^2,
    )
end

# From Section 1.2 of "Example Programs for ARKode v4.4.0" by D. R. Reynolds
function ark_analytic_nonlin_test_cts(::Type{FT}) where {FT}
    ClimaIntegratorTestCase(;
        test_name = "ark_analytic_nonlin",
        linear_implicit = false,
        t_end = FT(10),
        Y₀ = FT[0],
        analytic_sol = (t) -> [log(t^2 / 2 + t + 1)],
        tendency! = (Yₜ, Y, _, t) -> Yₜ .= (t + 1) .* exp.(.-Y),
        increment! = (Y⁺, Y, _, t, Δt) -> Y⁺ .+= Δt .* ((t + 1) .* exp.(.-Y)),
        Wfact! = (W, Y, _, Δt, t) -> W .= (-Δt * (t + 1) .* exp.(.-Y) .- 1),
        tgrad! = (∂Y∂t, Y, _, t) -> ∂Y∂t .= exp.(.-Y),
    )
end

# From Section 5.1 of "Example Programs for ARKode v4.4.0" by D. R. Reynolds
function ark_analytic_sys_test_cts(::Type{FT}) where {FT}
    λ = FT(-100) # increase magnitude for more stiffness
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
        increment! = (Y⁺, Y, _, t, Δt) -> mul!(Y⁺, A, Y, Δt, 1),
        Wfact! = (W, Y, _, Δt, t) -> W .= Δt .* A .- I,
    )
end

function ark_onewaycouple_mri_test_cts(::Type{FT}) where {FT}
    L = FT[0 -50 0; 50 0 0; 1 1 -1]
    I = LinearAlgebra.I(3)
    ClimaIntegratorTestCase(;
        test_name = "ark_onewaycouple_mri",
        linear_implicit = true,
        t_end = FT(1),
        Y₀ = FT[1, 0, 2],
        analytic_sol = (t) -> [
            cos(50 * t),
            sin(50 * t),
            5051 / 2501 * exp(-t) - 49 / 2501 * cos(50 * t) + 51 / 2501 * sin(50 * t),
        ],
        tendency! = (Yₜ, Y, _, t) -> mul!(Yₜ, L, Y),
        increment! = (Y⁺, Y, _, t, Δt) -> mul!(Y⁺, L, Y, Δt, 1),
        Wfact! = (W, Y, _, Δt, t) -> W .= Δt .* L .- I,
    )
end

#=
Due to conservation of (heat) energy, we know that
    c * ρ * ∂u/∂t = k * Δu + q,
where u is the temperature (K), c is the specific heat capacity (J/kg/K), ρ is
the density (kg/m^3), k is the thermal conductivity (W/m/K), and q is the rate
at which heat energy is added/removed (W/m^3).

We can simplify this PDE to
    ∂u/∂t = α * Δu + f,
where α = k/c/ρ is the thermal diffusivity (m^2/s) and f = q/c/ρ is the rate at
which heat energy is added/removed in units of temperature (K/s).

We will solve this PDE for u(x, y, t) over the domain (x, y) ∈ [0, l] × [0, l]
and t ≥ 0. For simplicity, we will use periodic boundary conditions:
    u(0, y, t) = u(l, y, t),
    u(x, 0, t) = u(x, l, t),
    ∇u(0, y, t) = ∇u(l, y, t), and
    ∇u(x, 0, t) = ∇u(x, l, t).
Also, for simplicity, we will assume that α is a constant.

Suppose that
    f = 0 and
    u(x, y, 0) = u₀(x, y).
The general solution to the PDE (obtained with separation of variables) is then
    u(x, y, t) =
        ∑_{n = 0}^∞ ∑_{m = 0}^∞ exp(-λₙₘ * t) * (
            φᶜᶜₙₘ(x, y) * ⟨φᶜᶜₙₘ(x, y), u₀(x, y)⟩ / ⟨φᶜᶜₙₘ(x, y), φᶜᶜₙₘ(x, y)⟩ +
            φᶜˢₙₘ(x, y) * ⟨φᶜˢₙₘ(x, y), u₀(x, y)⟩ / ⟨φᶜˢₙₘ(x, y), φᶜˢₙₘ(x, y)⟩ +
            φˢᶜₙₘ(x, y) * ⟨φˢᶜₙₘ(x, y), u₀(x, y)⟩ / ⟨φˢᶜₙₘ(x, y), φˢᶜₙₘ(x, y)⟩ +
            φˢˢₙₘ(x, y) * ⟨φˢˢₙₘ(x, y), u₀(x, y)⟩ / ⟨φˢˢₙₘ(x, y), φˢˢₙₘ(x, y)⟩
        ), where
    φᶜᶜₙₘ(x, y) = cos(2 * π * n * x / l) * cos(2 * π * m * y / l),
    φᶜˢₙₘ(x, y) = cos(2 * π * n * x / l) * sin(2 * π * m * y / l),
    φˢᶜₙₘ(x, y) = sin(2 * π * n * x / l) * cos(2 * π * m * y / l),
    φˢˢₙₘ(x, y) = sin(2 * π * n * x / l) * sin(2 * π * m * y / l), and
    λₙₘ = (2 * π / l)^2 * (n^2 + m^2) * α.
Note that the inner product of two functions g(x, y) and h(x, y) is defined as
    ⟨g(x, y), h(x, y)⟩ = ∫_0^l ∫_0^l g(x, y) h(x, y) dx dy.
When n = 0 or m = 0, some of the inner product denominators above are 0, but
this doesn't actually matter because the corresponding numerators are also 0 and
those terms can just be ignored.

So, the solution operator for the homogeneous PDE (with f = 0) is
    F(u₀)(x, y, t) =
        ∑_{n = 0}^∞ ∑_{m = 0}^∞ exp(-λₙₘ * t) * (
            φᶜᶜₙₘ(x, y) * ⟨φᶜᶜₙₘ(x, y), u₀(x, y)⟩ / ⟨φᶜᶜₙₘ(x, y), φᶜᶜₙₘ(x, y)⟩ +
            φᶜˢₙₘ(x, y) * ⟨φᶜˢₙₘ(x, y), u₀(x, y)⟩ / ⟨φᶜˢₙₘ(x, y), φᶜˢₙₘ(x, y)⟩ +
            φˢᶜₙₘ(x, y) * ⟨φˢᶜₙₘ(x, y), u₀(x, y)⟩ / ⟨φˢᶜₙₘ(x, y), φˢᶜₙₘ(x, y)⟩ +
            φˢˢₙₘ(x, y) * ⟨φˢˢₙₘ(x, y), u₀(x, y)⟩ / ⟨φˢˢₙₘ(x, y), φˢˢₙₘ(x, y)⟩
        ).
We can express the initial condition of our PDE using in terms of its Fourier
series as
    u₀(x, y) =
        ∑_{n = 0}^∞ ∑_{m = 0}^∞ (
            u₀ᶜᶜₙₘ * φᶜᶜₙₘ(x, y) +
            u₀ᶜˢₙₘ * φᶜˢₙₘ(x, y) +
            u₀ˢᶜₙₘ * φˢᶜₙₘ(x, y) +
            u₀ˢˢₙₘ * φˢˢₙₘ(x, y)
        ).
Now, consider the inhomogeneous PDE for which
    f(x, y, t) = f̂(t)(x, y) =
        ∑_{n = 0}^∞ ∑_{m = 0}^∞ exp(-λ′ₙₘ * t) * (
            f₀ᶜᶜₙₘ * φᶜᶜₙₘ(x, y) +
            f₀ᶜˢₙₘ * φᶜˢₙₘ(x, y) +
            f₀ˢᶜₙₘ * φˢᶜₙₘ(x, y) +
            f₀ˢˢₙₘ * φˢˢₙₘ(x, y)
        ).
(Note that, if we allow the time-dependence to have a form that is not
exponential, the resulting solution will contain non-elementary integrals.)
Duhamel's formula tells us that the solution to the inhomogeneous PDE is
    u(x, y, t) = F(u₀)(x, y, t) + ∫_0^t F(f̂(τ))(x, y, t - τ) dτ =
        ∑_{n = 0}^∞ ∑_{m = 0}^∞ exp(-λₙₘ * t) * (
            u₀ᶜᶜₙₘ * φᶜᶜₙₘ(x, y) +
            u₀ᶜˢₙₘ * φᶜˢₙₘ(x, y) +
            u₀ˢᶜₙₘ * φˢᶜₙₘ(x, y) +
            u₀ˢˢₙₘ * φˢˢₙₘ(x, y)
        ) +
        ∫_0^t ∑_{n = 0}^∞ ∑_{m = 0}^∞ (
            exp(-λ′ₙₘ * τ - λₙₘ * (t - τ)) * (
                f₀ᶜᶜₙₘ * φᶜᶜₙₘ(x, y) +
                f₀ᶜˢₙₘ * φᶜˢₙₘ(x, y) +
                f₀ˢᶜₙₘ * φˢᶜₙₘ(x, y) +
                f₀ˢˢₙₘ * φˢˢₙₘ(x, y)
            )
        ) dτ =
        ∑_{n = 0}^∞ ∑_{m = 0}^∞ exp(-λₙₘ * t) * (
            u₀ᶜᶜₙₘ * φᶜᶜₙₘ(x, y) +
            u₀ᶜˢₙₘ * φᶜˢₙₘ(x, y) +
            u₀ˢᶜₙₘ * φˢᶜₙₘ(x, y) +
            u₀ˢˢₙₘ * φˢˢₙₘ(x, y)
        ) +
        ∑_{n = 0}^∞ ∑_{m = 0}^∞ (
            exp(-λₙₘ * t) / (λₙₘ - λ′ₙₘ) * (exp((λₙₘ - λ′ₙₘ) * t) - 1)
        ) * (
            f₀ᶜᶜₙₘ * φᶜᶜₙₘ(x, y) +
            f₀ᶜˢₙₘ * φᶜˢₙₘ(x, y) +
            f₀ˢᶜₙₘ * φˢᶜₙₘ(x, y) +
            f₀ˢˢₙₘ * φˢˢₙₘ(x, y)
        ).
If we let λ′ₙₘ = λₙₘ + Δλₙₘ, this simplifies to
    u(x, y, t) =
        ∑_{n = 0}^∞ ∑_{m = 0}^∞ exp(-λₙₘ * t) * (
            u₀ᶜᶜₙₘ * φᶜᶜₙₘ(x, y) +
            u₀ᶜˢₙₘ * φᶜˢₙₘ(x, y) +
            u₀ˢᶜₙₘ * φˢᶜₙₘ(x, y) +
            u₀ˢˢₙₘ * φˢˢₙₘ(x, y)
        ) +
        ∑_{n = 0}^∞ ∑_{m = 0}^∞ exp(-λₙₘ * t) / Δλₙₘ * (1 - exp(-Δλₙₘ * t)) * (
            f₀ᶜᶜₙₘ * φᶜᶜₙₘ(x, y) +
            f₀ᶜˢₙₘ * φᶜˢₙₘ(x, y) +
            f₀ˢᶜₙₘ * φˢᶜₙₘ(x, y) +
            f₀ˢˢₙₘ * φˢˢₙₘ(x, y)
        ).

For the test case below, we will only use the φˢˢₙₘ eigenfunction for specific
values of n and m. In other words, we will pick some constants n, m, u₀, f₀, and
Δλ, and we will set
    u(x, y, 0) = u₀ * φˢˢₙₘ(x, y) and
    f(x, y, t) = f₀ * exp(-(λₙₘ + Δλ) * t) * φˢˢₙₘ(x, y).
We should then end up with the solution
    u(x, y, t) =
        (u₀ + f₀ / Δλ * (1 - exp(-Δλ * t))) * exp(-λₙₘ * t) * φˢˢₙₘ(x, y).

In addition, we will use nondimensionalization to replace our variables with
    x̂ = x / l,
    ŷ = y / l,
    t̂ = t / (l^2 / α),
    û(x̂, ŷ, t̂) = u(x, y, t) / u₀,
    f̂(x̂, ŷ, t̂) = f(x, y, t) / (u₀ / (l^2 / α)).
Note that this converts the time t into the "Fourier number" α * t / l^2.
We will then define the nondimensionalized constants
    λ̂ₙₘ = λₙₘ * l^2 / α = (2 * π)^2 * (n^2 + m^2),
    Δλ̂ = Δλ * l^2 / α, and
    f̂₀ = f₀ / (u₀ / (l^2 / α))
We will also rewrite the eigenfunction in terms of the new variables as
    φ̂ˢˢₙₘ(x̂, ŷ) = φˢˢₙₘ(x̂ * l, ŷ * l) = sin(2 * π * n * x̂) * sin(2 * π * m * ŷ).
Our simplified PDE then becomes
    ∂û/∂t̂ = Δû + f̂, where
    û(x̂, ŷ, 0) = φ̂ˢˢₙₘ(x̂, ŷ) and
    f̂(x̂, ŷ, t̂) = f̂₀ * exp(-(λ̂ₙₘ + Δλ̂) * t̂) * φ̂ˢˢₙₘ(x̂, ŷ).
Our solution then becomes
    û(x̂, ŷ, t̂) =
        (1 + f̂₀ / Δλ̂ * (1 - exp(-Δλ̂ * t̂))) * exp(-λ̂ₙₘ * t̂) * φ̂ˢˢₙₘ(x̂, ŷ).
In order to improve readability, we will drop the hats from all variable names.
=#
function climacore_2Dheat_test_cts(::Type{FT}) where {FT}
    dss_tendency = true

    n_elem_x = 2
    n_elem_y = 2
    n_poly = 2
    n_x = 1 # denoted by n above
    n_y = 1 # denoted by m above
    f_0 = FT(0) # denoted by f̂₀ above
    Δλ = FT(1) # denoted by Δλ̂ above
    t_end = FT(0.05) # denoted by t̂ above

    domain = ClimaCore.Domains.RectangleDomain(
        ClimaCore.Domains.IntervalDomain(
            ClimaCore.Geometry.XPoint(FT(0)),
            ClimaCore.Geometry.XPoint(FT(1)),
            periodic = true,
        ),
        ClimaCore.Domains.IntervalDomain(
            ClimaCore.Geometry.YPoint(FT(0)),
            ClimaCore.Geometry.YPoint(FT(1)),
            periodic = true,
        ),
    )
    mesh = ClimaCore.Meshes.RectilinearMesh(domain, n_elem_x, n_elem_y)
    topology = ClimaCore.Topologies.Topology2D(mesh)
    quadrature = ClimaCore.Spaces.Quadratures.GLL{n_poly + 1}()
    space = ClimaCore.Spaces.SpectralElementSpace2D(topology, quadrature)
    (; x, y) = ClimaCore.Fields.coordinate_field(space)

    λ = (2 * FT(π))^2 * (n_x^2 + n_y^2)
    φ_sin_sin = @. sin(2 * FT(π) * n_x * x) * sin(2 * FT(π) * n_y * y)

    init_state = ClimaCore.Fields.FieldVector(; u = φ_sin_sin)

    wdiv = ClimaCore.Operators.WeakDivergence()
    grad = ClimaCore.Operators.Gradient()
    function T_exp!(tendency, state, _, t)
        @. tendency.u = wdiv(grad(state.u)) + f_0 * exp(-(λ + Δλ) * t) * φ_sin_sin
        dss_tendency && ClimaCore.Spaces.weighted_dss!(tendency.u)
    end

    function dss!(state, _, t)
        dss_tendency || ClimaCore.Spaces.weighted_dss!(state.u)
    end

    function analytic_sol(t)
        u = @. (1 + f_0 / Δλ * (1 - exp(-Δλ * t))) * exp(-λ * t) * φ_sin_sin
        return ClimaCore.Fields.FieldVector(; u)
    end

    tendency_func = ClimaODEFunction(; T_exp!, dss!)
    split_tendency_func = tendency_func
    make_prob(func) = ODEProblem(func, init_state, (FT(0), t_end), nothing)
    IntegratorTestCase(
        "Horizontal Heat Equation",
        false,
        t_end,
        analytic_sol,
        make_prob(tendency_func),
        nothing,
        make_prob(split_tendency_func),
        nothing,
    )
end
function climacore_1Dheat_test_cts(::Type{FT}) where {FT}
    n_elem_z = 10
    n_z = 1
    f_0 = FT(0) # denoted by f̂₀ above
    Δλ = FT(1) # denoted by Δλ̂ above
    t_end = FT(0.1) # denoted by t̂ above

    domain = ClimaCore.Domains.IntervalDomain(
        ClimaCore.Geometry.ZPoint(FT(0)),
        ClimaCore.Geometry.ZPoint(FT(1)),
        boundary_names = (:bottom, :top),
    )
    mesh = ClimaCore.Meshes.IntervalMesh(domain, nelems = n_elem_z)
    space = ClimaCore.Spaces.FaceFiniteDifferenceSpace(mesh)
    (; z) = ClimaCore.Fields.coordinate_field(space)

    λ = (2 * FT(π) * n_z)^2
    φ_sin = @. sin(2 * FT(π) * n_z * z)

    init_state = ClimaCore.Fields.FieldVector(; u = φ_sin)

    div = ClimaCore.Operators.DivergenceC2F(;
        bottom = ClimaCore.Operators.SetDivergence(FT(0)),
        top = ClimaCore.Operators.SetDivergence(FT(0)),
    )
    grad = ClimaCore.Operators.GradientF2C()
    function T_exp!(tendency, state, _, t)
        @. tendency.u = div(grad(state.u)) + f_0 * exp(-(λ + Δλ) * t) * φ_sin
    end

    function analytic_sol(t)
        u = @. (1 + f_0 / Δλ * (1 - exp(-Δλ * t))) * exp(-λ * t) * φ_sin
        return ClimaCore.Fields.FieldVector(; u)
    end

    tendency_func = ClimaODEFunction(; T_exp!)
    split_tendency_func = tendency_func
    make_prob(func) = ODEProblem(func, init_state, (FT(0), t_end), nothing)
    IntegratorTestCase(
        "Vertical Heat Equation",
        false,
        t_end,
        analytic_sol,
        make_prob(tendency_func),
        nothing,
        make_prob(split_tendency_func),
        nothing,
    )
end
