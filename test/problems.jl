using DiffEqBase, ClimaTimeSteppers, LinearAlgebra, StaticArrays
using ClimaCore
using ClimaComms

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
Test problem (4.2) from [RobertsSarsharSandu2018arxiv](@cite)

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

struct IntegratorTestCase{FT, A, P, SP}
    test_name::String
    linear_implicit::Bool
    t_end::FT
    analytic_sol::A
    prob::P
    split_prob::SP
end

function IntegratorTestCase(;
    test_name,
    linear_implicit,
    t_end,
    Y₀,
    analytic_sol,
    tendency!,
    implicit_tendency! = nothing,
    explicit_tendency! = nothing,
    Wfact!,
    tgrad! = nothing,
)
    FT = typeof(t_end)
    jac_prototype = Matrix{FT}(undef, length(Y₀), length(Y₀))
    func_args = (; jac_prototype, Wfact = Wfact!, tgrad = tgrad!)
    tendency_func = ODEFunction(tendency!; func_args...)
    split_tendency_func = if isnothing(implicit_tendency!) # assume that related args are also nothing
        no_tendency!(Yₜ, Y, _, t) = Yₜ .= 0
        SplitFunction(tendency_func, no_tendency!)
    else
        SplitFunction(ODEFunction(implicit_tendency!; func_args...), explicit_tendency!)
    end
    make_prob(func) = ODEProblem(func, Y₀, (FT(0), t_end), nothing)
    IntegratorTestCase(
        test_name,
        linear_implicit,
        t_end,
        analytic_sol,
        make_prob(tendency_func),
        make_prob(split_tendency_func),
    )
end

function ClimaIntegratorTestCase(;
    test_name,
    linear_implicit,
    t_end,
    Y₀,
    analytic_sol,
    tendency!,
    implicit_tendency! = nothing,
    explicit_tendency! = nothing,
    Wfact!,
    tgrad! = nothing,
)
    FT = typeof(t_end)
    jac_prototype = Matrix{FT}(undef, length(Y₀), length(Y₀))
    func_args = (; jac_prototype, Wfact = Wfact!, tgrad = tgrad!)
    tendency_func = ClimaODEFunction(; T_imp! = ODEFunction(tendency!; func_args...))

    T_imp! = if isnothing(implicit_tendency!)
        # assume that related args are also nothing
        ODEFunction(tendency!; func_args...)
    else
        ODEFunction(implicit_tendency!; func_args...)
    end
    split_tendency_func = ClimaODEFunction(; T_exp! = explicit_tendency!, T_imp!)
    make_prob(func) = ODEProblem(func, Y₀, (FT(0), t_end), nothing)
    IntegratorTestCase(
        test_name,
        linear_implicit,
        t_end,
        analytic_sol,
        make_prob(tendency_func),
        make_prob(split_tendency_func),
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
        implicit_tendency! = (Yₜ, Y, _, t) -> Yₜ .= λ .* Y,
        explicit_tendency! = (Yₜ, Y, _, t) -> Yₜ .= source(t),
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
        Wfact! = (W, Y, _, Δt, t) -> W .= Δt .* A .- I,
    )
end

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
        Wfact! = (W, Y, _, Δt, t) -> W .= Δt .* L .- I,
    )
end

"""
    climacore_2Dheat_test_cts(::Type{<:AbstractFloat})

2D diffusion test problem. See [`2D diffusion problem`](@ref) for more details.
"""
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
        state = similar(init_state)
        @. state.u = (1 + f_0 / Δλ * (1 - exp(-Δλ * t))) * exp(-λ * t) * φ_sin_sin
        return state
    end

    tendency_func = ClimaODEFunction(; T_exp!, dss!)
    split_tendency_func = tendency_func
    make_prob(func) = ODEProblem(func, init_state, (FT(0), t_end), nothing)
    IntegratorTestCase(
        "2D Heat Equation",
        false,
        t_end,
        analytic_sol,
        make_prob(tendency_func),
        make_prob(split_tendency_func),
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
        state = similar(init_state)
        @. state.u = (1 + f_0 / Δλ * (1 - exp(-Δλ * t))) * exp(-λ * t) * φ_sin
        return state
    end

    tendency_func = ClimaODEFunction(; T_exp!)
    split_tendency_func = tendency_func
    make_prob(func) = ODEProblem(func, init_state, (FT(0), t_end), nothing)
    IntegratorTestCase(
        "1D Heat Equation",
        false,
        t_end,
        analytic_sol,
        make_prob(tendency_func),
        make_prob(split_tendency_func),
    )
end

function all_test_cases(::Type{FT}) where {FT}
    return [
        ark_analytic_nonlin_test_cts(FT),
        ark_analytic_sys_test_cts(FT),
        ark_analytic_test_cts(FT),
        onewaycouple_mri_test_cts(FT),
        climacore_2Dheat_test_cts(FT),
        climacore_1Dheat_test_cts(FT),
    ]
end
