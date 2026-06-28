export SSPKnoth, RosenbrockAlgorithm
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

!!! note
    The diagonal of `Γ` must be constant (all `Γ[i, i]` equal): the stepper uses
    a single `dtγ = dt * Γ[1, 1]` for every stage. The constructor asserts this.

# Fields
- `A`: transformed coefficient matrix ``A = α Γ^{-1}``.
- `α`: original time-dependent coefficient matrix.
- `C`: stepping matrix ``C = \\mathrm{diag}(Γ^{-1}) - Γ^{-1}``.
- `Γ`: substage contribution matrix.
- `m`: weight vector ``m = b Γ^{-1}``, used to compute the increments `k`.
"""
struct RosenbrockTableau{N, SM <: SMatrix{N, N}, SM1 <: SMatrix{N, 1}}
    A::SM
    α::SM
    C::SM
    Γ::SM
    m::SM1
end
n_stages(::RosenbrockTableau{N}) where {N} = N

function RosenbrockTableau(α::SMatrix{N, N}, Γ::SMatrix{N, N}, b::SMatrix{1, N}) where {N}
    # `step_u!` uses a single `dtγ = dt * Γ[1, 1]` for the Wfact and the stage
    # scaling of every stage, which is only correct when the diagonal of Γ is
    # constant. Reject tableaux that violate this rather than silently
    # producing wrong results on stages 2..N.
    @assert allequal(diag(Γ)) string(
        "RosenbrockAlgorithm requires a constant diagonal Γ (all Γ[i, i] equal); ",
        "got diag(Γ) = $(diag(Γ)).",
    )
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
    RosenbrockAlgorithm(name::RosenbrockAlgorithmName)
    RosenbrockAlgorithm(tableau::RosenbrockTableau)

A Rosenbrock-type ODE algorithm. The implicit system at each stage is a single
linear solve (no Newton iteration), making Rosenbrock methods cheaper per step
than fully implicit IMEX methods when updating the Jacobian is inexpensive.

The preferred constructor takes an algorithm name (e.g. [`SSPKnoth`](@ref)); a
raw [`RosenbrockTableau`](@ref) may also be passed.

# Example
```julia
alg = RosenbrockAlgorithm(SSPKnoth())
```
"""
struct RosenbrockAlgorithm{T <: RosenbrockTableau} <:
       ClimaTimeSteppers.TimeSteppingAlgorithm
    tableau::T
end

RosenbrockAlgorithm(name::RosenbrockAlgorithmName) = RosenbrockAlgorithm(tableau(name))

"""
    RosenbrockCache{Nstages, A, WT}

Pre-allocated workspace for a Rosenbrock-type method.

`Nstages` is the number of stages, `A` is the type of the evolved state
(e.g., a `ClimaCore.FieldVector`), and `WT` is the type of the Jacobian
operator (`Wfact`).

# Fields
- `U`: preallocated space for the stage state.
- `fU`: preallocated space for the total tendency.
- `fU_imp`: preallocated space for the implicit tendency.
- `fU_exp`: preallocated space for the explicit tendency.
- `fU_lim`: preallocated space for the limited tendency.
- `k`: stage increments (one per stage).
- `W`: preallocated Jacobian operator ``W = dtγ\\, J - I``.
- `∂Y∂t`: preallocated space for the explicit time derivative.
"""
struct RosenbrockCache{Nstages, A, WT}
    U::A
    fU::A
    fU_imp::A
    fU_exp::A
    fU_lim::A
    k::NTuple{Nstages, A}
    W::WT
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

Take one step with the Rosenbrock method using the given `cache`.

!!! note
    The `tgrad` correction accounts only for the explicit time dependence of the
    *implicit* tendency `T_imp!` (`tgrad` lives on the [`ODEFunction`](@ref)
    wrapping `T_imp!`). Explicit time dependence of `T_exp!` is not included.

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

    (; cache!, dss!, constrain_state!) = f
    (; update_cache, update_constrain_state) = f

    # TODO: This is only valid when Γ[i, i] is constant, otherwise we have to
    # move this in the for loop
    @inbounds dtγ = float(dt) * Γ[1, 1]

    if !isnothing(T_imp!)
        Wfact! = int.sol.prob.f.T_imp!.Wfact
        Wfact!(W, u, p, dtγ, t)
    end
    # W is constant across stages, so a dense W (CPU testing only) is factored
    # once here rather than re-factoring it every stage. Custom operators and GPU
    # arrays support `ldiv!` directly and skip the factorization.
    W_factorized = (!isnothing(T_imp!) && W isa DenseMatrix) ? lu(W) : W

    if !isnothing(tgrad!)
        tgrad!(∂Y∂t, u, p, t)
    end

    @inbounds for i in 1:Nstages
        # Reset tendency
        fill!(fU, 0)

        # Sum over the tableau rows with explicit loops rather than slices
        # to avoid heap allocations
        αi = zero(eltype(α))
        γi = zero(eltype(Γ))
        for j in 1:(i - 1)
            αi += α[i, j]
        end
        for j in 1:i
            γi += Γ[i, j]
        end

        U .= u
        for j in 1:(i - 1)
            U .+= A[i, j] .* k[j]
        end

        # We apply DSS and update the cache on every stage but the first,
        # and at the end of each timestep. Since the first stage is
        # unchanged from the end of the previous timestep, this order of
        # operations ensures that the state is always continuous and
        # consistent with the cache, including between timesteps.
        # State is ready for tendency evaluation right after assembly + DSS;
        # fire `EndOfStageSignal` (subtype of `WithDSS`).
        if i != 1
            dss!(U, p, t + αi * dt)
            needs_update!(update_constrain_state, EndOfStageSignal()) &&
                constrain_state!(U, p, t + αi * dt)
            needs_update!(update_cache, EndOfStageSignal()) && cache!(U, p, t + αi * dt)
        end

        if !isnothing(T_imp!)
            T_imp!(fU_imp, U, p, t + αi * dt)
            fU .+= fU_imp
        end

        if !isnothing(T_exp_T_lim!)
            T_exp_T_lim!(fU_exp, fU_lim, U, p, t + αi * dt)
            has_T_exp(f) && (fU .+= fU_exp)
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
            ldiv!(k[i], W_factorized, fU)
        else
            k[i] .= .-fU
        end
    end

    @inbounds for i in 1:Nstages
        u .+= m[i] .* k[i]
    end

    # End of step: `EndOfStepSignal <: EndOfStage <: WithDSS` fires all
    # three handler families with a single signal.
    dss!(u, p, t + dt)
    needs_update!(update_constrain_state, EndOfStepSignal()) &&
        constrain_state!(u, p, t + dt)
    needs_update!(update_cache, EndOfStepSignal()) && cache!(u, p, t + dt)
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
