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
    # `γ === nothing` (non-SDIRK paired with a per-timestep update signal) is
    # already rejected by `check_sdirk_compat` at cache construction.
    T_imp!.Wfact(jacobian, u, p, dt * γ, t)
    return nothing
end

"""
    val_to_int(::Val{N}) -> N

Return the static integer `N` from a `Val{N}` wrapper.
"""
@inline val_to_int(::Val{N}) where {N} = N

"""
    step_stages!(integrator, cache, v_s)

Update the Jacobian and run all stages for one timestep.

# Optimization opportunity (GPU)

For ESDIRK methods (`a_imp[1, 1] == 0`) the Jacobian is not consumed until
stage 2.  On GPU backends with multiple compute streams the Jacobian assembly
(`Wfact`) could therefore be launched on a secondary stream in parallel with
stage 1, then synchronized before stage 2 begins.  This is not currently
implemented because profiling showed negligible benefit for the Jacobian sizes
used in ClimaAtmos, but the structure here (update, then stages) is arranged
so that such an overlap can be added without changing the call sites.
"""
function step_stages!(integrator, cache, v_s)
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

# Whether an `update_j` handler would fire on the `NewTimeStep` signal. Used at
# cache construction to detect a non-SDIRK tableau paired with a per-timestep
# Jacobian refresh, which is mathematically inconsistent (see `sdirk_error`).
fires_on_new_timestep(::Any) = false
fires_on_new_timestep(::UpdateEvery{NewTimeStep}) = true
fires_on_new_timestep(::UpdateEveryN{NewTimeStep}) = true
fires_on_new_timestep(::UpdateEveryDt) = true

# Throw `sdirk_error` at construction if the algorithm's Jacobian-update policy
# is incompatible with the implicit tableau's diagonal structure.
function check_sdirk_compat(alg, γ)
    isnothing(γ) || return nothing
    isnothing(alg.newtons_method) && return nothing
    fires_on_new_timestep(alg.newtons_method.update_j) && sdirk_error(alg.name)
    return nothing
end
