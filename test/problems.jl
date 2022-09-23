using DiffEqBase, ClimaTimeSteppers, LinearAlgebra, StaticArrays

if !@isdefined(ArrayType)
    ArrayType = Array
end

const_prob = ODEProblem{true}(
    (du,u,p,t,α=true,β=false) -> (du .= α .* p .+ β .* du),
    [0.0],(0.0,1.0),2.0)
const_prob_fe = ODEProblem(
        ForwardEulerODEFunction((un,u,p,t,dt) -> (un .= u .+ dt.* p)),
        [0.0],(0.0,1.0),2.0)
function const_sol(u0,p,t)
    u0 .+ p*t
end




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
linear_prob = IncrementingODEProblem{true}(
    (du,u,p,t,α=true,β=false) -> (du .= α .* p .* u .+ β .* du),
    [1/2],(0.0,1.0),1.01)
linear_prob_fe = ODEProblem(
    ForwardEulerODEFunction((un,u,p,t,dt) -> (un .= u .+ dt .* p .* u)),
    [1.0],(0.0,1.0),-0.2)
    
linear_prob_wfactt = ODEProblem(
        ODEFunction(
          (du,u,p,t,α=true,β=false) -> (du .= α .* p .* u .+ β .* du);
          jac_prototype=zeros(1,1),
          Wfact_t = (W,u,p,γ,t) -> (W[1,1]=1/γ-p),
        ),
        [1/2],(0.0,1.0),-0.2)

split_linear_prob_wfact_split = ODEProblem(
    SplitFunction(
        ODEFunction(
            (du,u,p,t) -> (du .= real(p) .* u);
            jac_prototype=zeros(ComplexF64,1,1),
            Wfact = (W,u,p,γ,t) -> (W[1,1]=γ*real(p)-1),
        ),
        ODEFunction((du, u, p, t) -> (du .= imag(p) * im .* u)),
    ),
    [1/2 + 0.0*im],(0.0,1.0),-0.2+0.1*im)

split_linear_prob_wfact_split_fe = ODEProblem(
    SplitFunction(
        ODEFunction(
            (du,u,p,t) -> (du .= real(p) .* u);
            jac_prototype=zeros(ComplexF64,1,1),
            Wfact = (W,u,p,γ,t) -> (W[1,1]=γ*real(p)-1),
        ),
        ForwardEulerODEFunction(
            (ux, u, p, t, dt) -> (ux .= ux .+ dt .* imag(p) * im .* u),
        ),
    ),
    [1/2 + 0.0*im],(0.0,1.0),-0.2+0.1*im)


# DiffEqProblemLibrary.jl uses the `analytic` argument to ODEFunction to store the exact solution
# IncrementingODEFunction doesn't have that
function linear_sol(u0,p,t)
    u0 .* exp(p*t)
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
sincos_prob = IncrementingODEProblem{true}(
    (du,u,p,t,α=true,β=false) -> (du[1] = α*p*u[2]+β*du[1]; du[2] = -α*p*u[1]+β*du[2]),
    [0.0,1.0], (0.0,1.0), 2.0)
sincos_prob_fe = ODEProblem(
    ForwardEulerODEFunction((un,u,p,t,dt) -> (un[1] = u[1] + dt*p*u[2]; un[2] = u[2]-dt*p*u[1])),
    [0.0,1.0], (0.0,1.0), 2.0)

function sincos_sol(u0,p,t)
    s,c = sincos(p*t)
    [c s; -s c]*u0
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
imex_autonomous_prob = SplitODEProblem(
    (du, u, p, t, α=true, β=false) -> (du .= α .* u        .+ β .* du),
    (du, u, p, t, α=true, β=false) -> (du .= α .* cos(t)/p .+ β .* du),
    ArrayType([0.5]), (0.0,1.0), 4.0)

function linsolve_direct(::Type{Val{:init}}, f, u0; kwargs...)
    _linsolve!(x, A, b, update_matrix = false; kwargs...) = x .= A \ b
end
multiply_direct!(b, A, x) = mul!(b, A, x)
set_Δtγ_direct!(A, Δtγ_new, Δtγ_old) =
    A .= (A .+ I(size(A, 1))) .* (Δtγ_new / Δtγ_old) .- I(size(A, 1))

imex_autonomous_prob_jac = ODEProblem(
        ODEFunction(
            (du, u, p, t, α=true, β=false) -> (du .= α .* (u .+ cos(t)/p) .+ β .* du),
            jac_prototype = zeros(1,1),
            Wfact = (W,u,p,γ,t) -> W[1,1] = 1-γ,
        ),
        ArrayType([0.5]), (0.0,1.0), 4.0)

function imex_autonomous_sol(u0,p,t)
    (exp(t)  + sin(t) - cos(t)) / 2p .+ exp(t) .* u0
end

imex_nonautonomous_prob = SplitODEProblem(
    (du, u, p, t, α=true, β=false) -> (du .= α .* cos(t) * u .+ β .* du),
    (du, u, p, t, α=true, β=false) -> (du .= α .* cos(t)/p   .+ β .* du),
    ArrayType([0.5]), (0.0,2.0), 4.0)

function imex_nonautonomous_sol(u0,p,t)
    (exp(sin(t)) .* (1 .+ p .* u0) .- 1) ./ p
end


"""
Test problem (4.2) from RobertsSarsharSandu2018arxiv
@article{RobertsSarsharSandu2018arxiv,
    title={Coupled Multirate Infinitesimal GARK Schemes for Stiff Systems with
            Multiple Time Scales},
    author={Roberts, Steven and Sarshar, Arash and Sandu, Adrian},
    journal={arXiv preprint arXiv:1812.00808},
    year={2019}
}

Note: The actual rates are all over the place with this test and passing largely
        depends on final dt size
"""
kpr_param = (
    ω = 100,
    λf = -10,
    λs = -1,
    ξ = 0.1,
    α = 1,
)

function kpr_rhs(Q,param,t)
    yf, ys = Q
    ω, λf, λs, ξ, α = param

    ηfs = ((1 - ξ) / α) * (λf - λs)
    ηsf = -ξ * α * (λf - λs)
    Ω = @SMatrix [
        λf ηfs
        ηsf λs
    ]

    g = @SVector [
        (-3 + yf^2 - cos(ω * t)) / 2yf,
        (-2 + ys^2 - cos(t)) / 2ys,
    ]
    h = @SVector [
        ω * sin(ω * t) / 2yf,
        sin(t) / 2ys,
    ]
    return Ω * g - h
end

function kpr_fast!(dQ, Q, param, t, α=1, β=0)
    dQf,_ = kpr_rhs(Q,param,t)
    dQ[1] = α*dQf + β*dQ[1]
    dQ[2] = β*dQ[2]
end

function kpr_slow!(dQ, Q, param, t, α=1, β=0)
    _,dQs = kpr_rhs(Q,param,t)
    dQ[1] = β*dQ[1]
    dQ[2] = α*dQs + β*dQ[2]
end

function kpr_sol(u0,param,t)
    #@assert u0 == [sqrt(4) sqrt(3)]
    ω, λf, λs, ξ, α = param
    return [
        sqrt(3 + cos(ω * t))
        sqrt(2 + cos(t))
    ]
end

kpr_multirate_prob = SplitODEProblem(
    IncrementingODEFunction{true}(kpr_fast!),
    IncrementingODEFunction{true}(kpr_slow!),
    [sqrt(4), sqrt(3)], (0.0, 5π/2), kpr_param,
)

kpr_singlerate_prob = IncrementingODEProblem{true}(
    (dQ,Q,param,t,α=1,β=0) -> (dQ .= α .* kpr_rhs(Q,param,t) .+ β .* dQ),
    [sqrt(4), sqrt(3)], (0.0, 5π/2), kpr_param,
)

struct IntegratorTestCase{FT, P, S, A}
    test_name::String
    linear_implicit::Bool
    t_end::FT
    probs::P
    split_probs::S
    analytic_sol::A
end

# From Section 1.1 of "Example Programs for ARKode v4.4.0" by D. R. Reynolds
ark_analytic = let
    FT = Float64
    λ = FT(-100) # increase magnitude for more stiffness
    Y₀ = FT[0]
    t_end = FT(10)

    source(t) = 1 / (1 + t^2) - λ * atan(t)
    tendency!(Yₜ, Y, _, t) = Yₜ .= λ .* Y .+ source(t)
    increment!(Y⁺, Y, _, t, Δt) = Y⁺ .+= Δt .* (λ .* Y .+ source(t))
    implicit_tendency!(Yₜ, Y, _, t) = Yₜ .= λ .* Y
    explicit_tendency!(Yₜ, Y, _, t) = Yₜ .= source(t)
    implicit_increment!(Y⁺, Y, _, t, Δt) = Y⁺ .+= (Δt * λ) .* Y
    explicit_increment!(Y⁺, Y, _, t, Δt) = Y⁺ .+= Δt * source(t)

    Wfact!(W, Y, _, Δt, t) = W .= Δt * λ - 1
    tgrad!(∂Y∂t, Y, _, t) = ∂Y∂t .= -(λ + 2 * t + λ * t^2) / (1 + t^2)^2
    analytic_sol(t) = [atan(t)]

    func_args = (; jac_prototype = Y₀, Wfact = Wfact!, tgrad = tgrad!)
    tendency_func = ODEFunction(tendency!; func_args...)
    increment_func = ForwardEulerODEFunction(increment!; func_args...)
    split_tendency_func = SplitFunction(
        ODEFunction(implicit_tendency!; func_args...),
        explicit_tendency!,
    )
    split_increment_func = SplitFunction(
        ForwardEulerODEFunction(implicit_increment!; func_args...),
        ForwardEulerODEFunction(explicit_increment!),
    )

    make_prob(func) = ODEProblem(func, Y₀, (FT(0), t_end), nothing)
    IntegratorTestCase(
        "ark_analytic",
        true,
        t_end,
        (make_prob(tendency_func), make_prob(increment_func)),
        (make_prob(split_tendency_func), make_prob(split_increment_func)),
        analytic_sol,
    )
end

# From Section 1.2 of "Example Programs for ARKode v4.4.0" by D. R. Reynolds
ark_analytic_nonlin = let
    FT = Float64
    Y₀ = FT[0]
    t_end = FT(10)

    tendency!(Yₜ, Y, _, t) = Yₜ .= (t + 1) .* exp.(.-Y)
    increment!(Y⁺, Y, _, t, Δt) = Y⁺ .+= Δt .* ((t + 1) .* exp.(.-Y))
    no_tendency!(Yₜ, Y, _, t) = Yₜ .= zero(FT)
    no_increment!(Y⁺, Y, _, t, Δt) = Y⁺

    Wfact!(W, Y, _, Δt, t) = W .= (-Δt * (t + 1) .* exp.(.-Y) .- 1)
    tgrad!(∂Y∂t, Y, _, t) = ∂Y∂t .= exp.(.-Y)
    analytic_sol(t) = [log(t^2 / 2 + t + 1)]

    func_args = (; jac_prototype = Y₀, Wfact = Wfact!, tgrad = tgrad!)
    tendency_func = ODEFunction(tendency!; func_args...)
    split_tendency_func = SplitFunction(tendency_func, no_tendency!)
    increment_func = ForwardEulerODEFunction(increment!; func_args...)
    split_increment_func =
        SplitFunction(increment_func, ForwardEulerODEFunction(no_increment!))

    make_prob(func) = ODEProblem(func, Y₀, (FT(0), t_end), nothing)
    IntegratorTestCase(
        "ark_analytic_nonlin",
        false,
        t_end,
        (make_prob(tendency_func), make_prob(increment_func)),
        (make_prob(split_tendency_func), make_prob(split_increment_func)),
        analytic_sol,
    )
end

# From Section 5.1 of "Example Programs for ARKode v4.4.0" by D. R. Reynolds
ark_analytic_sys = let
    FT = Float64
    λ = FT(-100) # increase magnitude for more stiffness
    V = FT[1 -1 1; -1 2 1; 0 -1 2]
    V⁻¹ = FT[5 1 -3; 2 2 -2; 1 1 1] / 4
    D = Diagonal(FT[-1/2, -1/10, λ])
    A = V * D * V⁻¹
    I = LinearAlgebra.I(3)
    Y₀ = FT[1, 1, 1]
    t_end = FT(1 / 20)

    tendency!(Yₜ, Y, _, t) = mul!(Yₜ, A, Y)
    increment!(Y⁺, Y, _, t, Δt) = mul!(Y⁺, A, Y, Δt, 1)
    no_tendency!(Yₜ, Y, _, t) = Yₜ .= zero(FT)
    no_increment!(Y⁺, Y, _, t, Δt) = Y⁺

    Wfact!(W, Y, _, Δt, t) = W .= Δt .* A .- I
    analytic_sol(t) = V * exp(D * t) * V⁻¹ * Y₀

    func_args = (; jac_prototype = A, Wfact = Wfact!)
    tendency_func = ODEFunction(tendency!; func_args...)
    split_tendency_func = SplitFunction(tendency_func, no_tendency!)
    increment_func = ForwardEulerODEFunction(increment!; func_args...)
    split_increment_func =
        SplitFunction(increment_func, ForwardEulerODEFunction(no_increment!))

    make_prob(func) = ODEProblem(func, Y₀, (FT(0), t_end), nothing)
    IntegratorTestCase(
        "ark_analytic_sys",
        true,
        t_end,
        (make_prob(tendency_func), make_prob(increment_func)),
        (make_prob(split_tendency_func), make_prob(split_increment_func)),
        analytic_sol,
    )
end
