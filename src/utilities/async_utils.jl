# Jacobian update logic and stage-dispatch helpers shared by IMEXARKCache
# and IMEXSSPRKCache.

"""
    maybe_update_jacobian!(T_imp!, newtons_method, cache, u, p, t, dt, γ, alg)

Synchronously update the Jacobian (Wfact) at the start of a timestep when the
`NewTimeStep` signal fires.  No-op when `T_imp!`, `newtons_method`, or the
Jacobian buffer is `nothing`, or when the update policy says no update is due.
"""
function maybe_update_jacobian!(
    T_imp!,
    newtons_method,
    newtons_method_cache,
    u,
    p,
    t,
    dt,
    γ,
    alg,
)
    isnothing(T_imp!) && return nothing
    isnothing(newtons_method) && return nothing
    jacobian = newtons_method_cache.j
    isnothing(jacobian) && return nothing
    needs_update!(newtons_method.update_j, NewTimeStep(t)) || return nothing
    γ isa Nothing && sdirk_error(alg.name)
    T_imp!.Wfact(jacobian, u, p, dt * γ, t)
    return nothing
end

"""
    val_to_int(::Val{N}) -> N

Unwrap a `Val{N}` to its static integer.  Used by the IMEX step loops to read
back the stage count from a `Val`-wrapped tableau dimension without introducing
a runtime type instability.
"""
@inline val_to_int(::Val{N}) where {N} = N

"""
    step_stages!(integrator, cache, a_imp, v_s)

Update the Jacobian and run all stages for one timestep.
"""
function step_stages!(integrator, cache, a_imp, v_s)
    (; u, p, t, dt, alg) = integrator
    (; T_imp!) = integrator.sol.prob.f
    (; newtons_method) = alg
    (; γ, newtons_method_cache) = cache

    maybe_update_jacobian!(
        T_imp!,
        newtons_method,
        newtons_method_cache,
        u, p, t, dt, γ, alg,
    )
    update_stage!(integrator, cache, ntuple(j -> Val(j), v_s))
    return nothing
end

# Internal helpers ------------------------------------------------------------

# Raised when an IMEX algorithm with non-uniform diagonal `a_imp` coefficients
# is used with the `NewTimeStep` update signal. Such tableaux require a fresh
# Jacobian per stage, not per timestep.
sdirk_error(name) = error("$(isnothing(name) ? "The given IMEXTableau" : name) \
                           has implicit stages with distinct coefficients (it \
                           is not SDIRK), and an update is required whenever a \
                           stage has a different coefficient from the previous \
                           stage. Do not update on the NewTimeStep signal when \
                           using $(isnothing(name) ? "this tableau" : name).")
