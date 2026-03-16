export SSPKnoth
using StaticArrays
import LinearAlgebra: ldiv!, diagm
import LinearAlgebra

abstract type RosenbrockAlgorithmName <: AbstractAlgorithmName end

"""
    RosenbrockTableau{N}

Tableau for an `N`-stage Rosenbrock method. Stores the transformed coefficients
``A = α Γ^{-1}``, ``C = \\mathrm{diag}(Γ^{-1}) - Γ^{-1}``, and
``m = b Γ^{-1}`` used in the actual computation, along with the original
``α`` and ``Γ``.

See the [Rosenbrock algorithm formulation](@ref rosenbrock-methods) for details.
"""
struct RosenbrockTableau{N, SM <: SMatrix{N, N}, SM1 <: SMatrix{N, 1}}
    """A = α Γ⁻¹"""
    A::SM
    """Tableau used for the time-dependent part"""
    α::SM
    """Stepping matrix. C = 1/diag(Γ) - Γ⁻¹"""
    C::SM
    """Substage contribution matrix"""
    Γ::SM
    """m = b Γ⁻¹, used to compute the increments k"""
    m::SM1
end
n_stages(::RosenbrockTableau{N}) where {N} = N

function RosenbrockTableau(α::SMatrix{N, N}, Γ::SMatrix{N, N}, b::SMatrix{1, N}) where {N}
    A = α / Γ
    invΓ = inv(Γ)
    diag_invΓ = SMatrix{N, N}(diagm([invΓ[i, i] for i in 1:N]))
    # C is diag(γ₁₁⁻¹, γ₂₂⁻¹, ...) - Γ⁻¹
    C = diag_invΓ .- inv(Γ)
    m = b / Γ
    m′ = convert(SMatrix{N, 1}, m) # Sometimes m is a SMatrix{1, N} matrix.
    SM = typeof(A)
    SM1 = typeof(m′)
    return RosenbrockTableau{N, SM, SM1}(A, α, C, Γ, m′)
end

"""
    RosenbrockAlgorithm(tableau)

A Rosenbrock-type ODE algorithm. The implicit system at each stage is a single
linear solve (no Newton iteration), making Rosenbrock methods cheaper per step
than fully implicit IMEX methods when the Jacobian is inexpensive.

# Arguments
- `tableau`: a [`RosenbrockTableau`](@ref) (e.g. `tableau(SSPKnoth())`)

# Example
```julia
import ClimaTimeSteppers as CTS

alg = CTS.RosenbrockAlgorithm(CTS.tableau(SSPKnoth()))
```
"""
struct RosenbrockAlgorithm{T <: RosenbrockTableau} <:
       ClimaTimeSteppers.TimeSteppingAlgorithm
    tableau::T
end

"""
    RosenbrockCache{N, A, WT}

Contains everything that is needed to run a Rosenbrock-type method.

- Nstages: number of stages,
- A: type of the evolved state (e.g., a ClimaCore.FieldVector),
- WT: type of the Jacobian (Wfact)
"""
struct RosenbrockCache{Nstages, A, WT}
    """Preallocated space for the state"""
    U::A

    """Preallocated space for the tendency"""
    fU::A

    """Preallocated space for the implicit contribution to the tendency"""
    fU_imp::A

    """Preallocated space for the explicit contribution to the tendency"""
    fU_exp::A

    """Preallocated space for the limited contribution to the tendency"""
    fU_lim::A

    """Contributions to the state for each stage"""
    k::NTuple{Nstages, A}

    """Preallocated space for the Wfact, dtγJ - 𝕀, or Wfact_t, 𝕀/dtγ - J, with J the Jacobian of the implicit tendency"""
    W::WT

    """Preallocated space for the explicit time derivative of the tendency"""
    ∂Y∂t::A
end

function init_cache(prob, alg::RosenbrockAlgorithm; kwargs...)
    Nstages = length(alg.tableau.m)
    U = zero(prob.u0)
    fU = zero(prob.u0)
    fU_imp = zero(prob.u0)
    fU_exp = zero(prob.u0)
    fU_lim = zero(prob.u0)
    ∂Y∂t = zero(prob.u0)
    k = ntuple(n -> zero(prob.u0), Nstages)
    if !isnothing(prob.f.T_imp!)
        W = prob.f.T_imp!.jac_prototype
    else
        W = nothing
    end
    return RosenbrockCache{Nstages, typeof(U), typeof(W)}(
        U,
        fU,
        fU_imp,
        fU_exp,
        fU_lim,
        k,
        W,
        ∂Y∂t,
    )
end

"""
    step_u!(int, cache::RosenbrockCache{Nstages})

Take one step with the Rosenbrock-method with the given `cache`.

Some choices are being made here. Most of these are empirically motivated and should be
revisited on different problems.
- We do not update dtγ across stages
- We apply DSS to the sum of the explicit and implicit tendency at all the stages but the last
- We apply DSS to incremented state (ie, after the final stage is applied)
"""
function step_u!(int, cache::RosenbrockCache{Nstages}) where {Nstages}
    (; m, Γ, A, α, C) = int.alg.tableau
    (; u, p, t, dt) = int
    (; W, U, fU, fU_imp, fU_exp, fU_lim, k, ∂Y∂t) = cache
    f = int.sol.prob.f
    T_imp! = f.T_imp!
    T_exp_T_lim! = f.T_exp_T_lim!
    tgrad! = isnothing(T_imp!) ? nothing : T_imp!.tgrad

    (; cache!, dss!) = f

    # TODO: This is only valid when Γ[i, i] is constant, otherwise we have to
    # move this in the for loop
    @inbounds dtγ = float(dt) * Γ[1, 1]

    if !isnothing(T_imp!)
        Wfact! = int.sol.prob.f.T_imp!.Wfact
        Wfact!(W, u, p, dtγ, t)
    end

    if !isnothing(tgrad!)
        tgrad!(∂Y∂t, u, p, t)
    end

    @inbounds for i in 1:Nstages
        # Reset tendency
        fill!(fU, 0)

        αi = sum(α[i, 1:(i - 1)]; init = zero(eltype(α)))
        γi = sum(Γ[i, 1:i]; init = zero(eltype(Γ)))

        U .= u
        for j in 1:(i - 1)
            U .+= A[i, j] .* k[j]
        end

        # We apply DSS and update the cache on every stage but the first,
        # and at the end of each timestep. Since the first stage is
        # unchanged from the end of the previous timestep, this order of
        # operations ensures that the state is always continuous and
        # consistent with the cache, including between timesteps.
        (i != 1) && dss!(U, p, t + αi * dt)
        (i != 1) && cache!(U, p, t + αi * dt)

        if !isnothing(T_imp!)
            T_imp!(fU_imp, U, p, t + αi * dt)
            fU .+= fU_imp
        end

        if !isnothing(T_exp_T_lim!)
            T_exp_T_lim!(fU_exp, fU_lim, U, p, t + αi * dt)
            fU .+= fU_exp
            has_T_lim(f) && (fU .+= fU_lim)
        end

        if !isnothing(tgrad!)
            fU .+= γi .* float(dt) .* ∂Y∂t
        end

        for j in 1:(i - 1)
            fU .+= (C[i, j] / float(dt)) .* k[j]
        end

        fU .*= -float(dtγ)

        if !isnothing(T_imp!)
            if W isa Matrix
                ldiv!(k[i], lu(W), fU)
            else
                ldiv!(k[i], W, fU)
            end
        else
            k[i] .= .-fU
        end
    end

    @inbounds for i in 1:Nstages
        u .+= m[i] .* k[i]
    end

    dss!(u, p, t + dt)
    cache!(u, p, t + dt)
    return nothing
end

"""
    SSPKnoth

A 3-stage, 2nd-order Rosenbrock method developed by Oswald Knoth.

Use with [`RosenbrockAlgorithm`](@ref):
```julia
import ClimaTimeSteppers as CTS
alg = CTS.RosenbrockAlgorithm(CTS.tableau(SSPKnoth()))
```
"""
struct SSPKnoth <: RosenbrockAlgorithmName end

function tableau(::SSPKnoth)
    N = 3
    α = @SMatrix [
        0 0 0
        1 0 0
        1/4 1/4 0
    ]
    b = @SMatrix [1 / 6 1 / 6 2 / 3]
    Γ = @SMatrix [
        1 0 0
        0 1 0
        -3/4 -3/4 1
    ]
    return RosenbrockTableau(α, Γ, b)
end
