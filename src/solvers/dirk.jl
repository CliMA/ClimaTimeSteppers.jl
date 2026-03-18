import LinearAlgebra

export DIRKAlgorithm

"""
    DIRKAlgorithm(tableau, newtons_method, [constraint])
    DIRKAlgorithm(name, newtons_method, [constraint])

Constructs an algorithm for (E)SDIRK methods using a single implicit/full RHS.
"""
struct DIRKAlgorithm{
    C <: AbstractAlgorithmConstraint,
    N <: Union{Nothing, AbstractAlgorithmName},
    T <: DIRKTableau,
    NM <: NewtonsMethod,
} <: DistributedODEAlgorithm
    constraint::C
    name::N
    tableau::T
    newtons_method::NM
end

DIRKAlgorithm(
    tableau::DIRKTableau,
    newtons_method::NewtonsMethod,
    constraint::AbstractAlgorithmConstraint = Unconstrained(),
) = DIRKAlgorithm(constraint, nothing, tableau, newtons_method)

DIRKAlgorithm(
    name::DIRKAlgorithmName,
    newtons_method::NewtonsMethod,
    constraint::AbstractAlgorithmConstraint = default_constraint(name),
) = DIRKAlgorithm(constraint, name, tableau(name), newtons_method)

################################################################################
# Solver implementation
################################################################################

struct DIRKCache{U, T, Γ, NMC}
    U::U
    T::T
    temp::U
    γ::Γ
    newtons_method_cache::NMC
end

function init_cache(prob, alg::DIRKAlgorithm{Unconstrained}; kwargs...)
    (; u0, f) = prob
    (; tableau, newtons_method) = alg
    (; a, b) = tableau

    s = length(b)
    inds = ntuple(i -> i, s)
    inds_T = filter(i -> !all(iszero, a[:, i]) || !iszero(b[i]), inds)

    U = zero(u0)
    temp = zero(u0)

    # Store stage tendencies only for those needed later (saves memory and
    # allows skipping unnecessary RHS evaluations).
    T = SparseContainer(map(i -> zero(u0), collect(1:length(inds_T))), inds_T)

    γs = unique(filter(!iszero, LinearAlgebra.diag(a)))
    γ = length(γs) == 1 ? γs[1] : nothing

    jac_prototype = has_jac(f.T_imp!) ? f.T_imp!.jac_prototype : nothing
    newtons_method_cache =
        isnothing(f.T_imp!) || isnothing(newtons_method) ? nothing :
        allocate_cache(newtons_method, u0, jac_prototype)

    return DIRKCache(
        U,
        T,
        temp,
        γ,
        newtons_method_cache,
    )
end

"""
    step_u!(integrator, cache::DIRKCache)

DIRK/ESDIRK advance using only `f.T_imp!` as the RHS.
"""
function step_u!(integrator, cache::DIRKCache)
    (; u, p, t, dt, alg) = integrator
    (; f) = integrator.sol.prob
    (; cache!, cache_imp!, dss!) = f
    (; tableau, newtons_method) = alg
    (; a, b, c) = tableau
    (; U, T, temp, γ, newtons_method_cache) = cache

    T_imp! = f.T_imp!
    @assert !isnothing(T_imp!) "DIRKAlgorithm requires f.T_imp! to be non-nothing"

    s = length(b)

    # Update Jacobian factorization if configured that way.
    if !isnothing(newtons_method)
        (; update_j) = newtons_method
        jacobian = newtons_method_cache.j
        if (!isnothing(jacobian)) && needs_update!(update_j, NewTimeStep(t))
            if γ isa Nothing
                # Non-constant diagonal => must update between stages (not
                # supported in this simplified DIRK path).
                error("DIRKAlgorithm requires constant diagonal (SDIRK/ESDIRK) when update_jacobian_every != \"solve\".")
            else
                # T_imp!.Wfact expects dtγ and time.
                T_imp!.Wfact(jacobian, u, p, dt * γ, t)
            end
        end
    end

    # Stage loop
    @. U = u
    for i in 1:s
        # Only evaluate/store stage i if it is needed by later stages or
        # by the final weights.
        need_Ti = !all(iszero, a[:, i]) || !iszero(b[i])
        need_Ti || continue

        t_i = t + dt * c[i]

        # U_i = u_n + dt * sum_{j<i} a[i,j] * T_j
        assign_fused_increment!(U, u, dt, a, T, Val(i))

        # Apply DSS + update caches for stage i (skip for stage 1, which is
        # already consistent with the end of the previous timestep).
        i != 1 && dss!(U, p, t_i)
        i != 1 && cache!(U, p, t_i)

        dtγ = float(dt) * a[i, i]

        if iszero(dtγ)
            # Explicit stage.
            T_imp!(T[i], U, p, t_i)
        else
            # Implicit stage.
            @. temp = U
            i != 1 && cache_imp!(U, p, t_i)

            @assert !isnothing(newtons_method)
            solve_implicit_equation!(
                U,
                temp,
                p,
                t_i,
                dtγ,
                T_imp!,
                newtons_method,
                newtons_method_cache,
                cache_imp!,
                dss!,
                cache!,
            )

            # Back out the tendency consistent with DSSed U.
            @. T[i] = (U - temp) / dtγ
        end
    end

    t_final = t + dt

    # u_{n+1} = u_n + dt * sum_i b[i] * T_i
    fused_increment!(u, dt, b, T, Val(s))

    dss!(u, p, t_final)
    cache!(u, p, t_final)

    return u
end

