"""
    init_jac_resources(u)

Allocate per-cache resources used to overlap Jacobian assembly with stage 1
evaluation. The default returns `nothing`; the CUDA extension returns a
`(stream, event)` tuple owned by the integrator's cache.
"""
init_jac_resources(u) = nothing

# Returns the Jacobian buffer that needs an update at this time step, or
# `nothing` if no update is needed. Stateful: consumes a signal from
# `newtons_method.update_j` via `needs_update!`, so calling this twice in a
# row will return `nothing` the second time. Not safe for inspection.
function claim_jacobian_update!(T_imp!, newtons_method, newtons_method_cache, t)
    isnothing(T_imp!) && return nothing
    isnothing(newtons_method) && return nothing
    jacobian = newtons_method_cache.j
    isnothing(jacobian) && return nothing
    needs_update!(newtons_method.update_j, NewTimeStep(t)) || return nothing
    return jacobian
end

"""
    maybe_update_jacobian!(T_imp!, newtons_method, cache, u, p, t, dt, γ, alg)

Synchronously update the Jacobian at the start of a timestep if the
`NewTimeStep` signal fires.
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
    jacobian = claim_jacobian_update!(T_imp!, newtons_method, newtons_method_cache, t)
    isnothing(jacobian) && return nothing
    γ isa Nothing && sdirk_error(alg.name)
    T_imp!.Wfact(jacobian, u, p, dt * γ, t)
    return nothing
end

"""
    async_update_jacobian!(T_imp!, newtons_method, cache, u, p, t, dt, γ, alg, jac_resources)

Like [`maybe_update_jacobian!`](@ref), but launches the Wfact computation
asynchronously when the backend supports it. Returns a synchronization token
to be passed to [`sync_jacobian_update!`](@ref) before the Jacobian is
consumed, or `nothing` if no update was launched.
"""
function async_update_jacobian!(
    T_imp!,
    newtons_method,
    newtons_method_cache,
    u,
    p,
    t,
    dt,
    γ,
    alg,
    jac_resources,
)
    jacobian = claim_jacobian_update!(T_imp!, newtons_method, newtons_method_cache, t)
    isnothing(jacobian) && return nothing
    γ isa Nothing && sdirk_error(alg.name)
    return async_Wfact!(T_imp!, jacobian, u, p, dt * γ, t, jac_resources)
end

"""
    async_Wfact!(T_imp!, jacobian, u, p, dtγ, t, jac_resources)

Compute the Wfact matrix. The CPU baseline runs synchronously and returns
`nothing`. The CUDA extension overrides this to run on a separate stream and
return a `CuEvent` for synchronization.

# Backend extension contract

A backend that wants stage-1 / Jacobian overlap must extend both
[`init_jac_resources`](@ref) (to allocate per-cache stream/event handles) and
`async_Wfact!` (dispatching on the backend's array type *and*
`jac_resources` type). Extending only `init_jac_resources` is harmless —
calls fall through to this synchronous method, which runs Wfact correctly on
the device's default stream but forgoes the overlap.
"""
function async_Wfact!(T_imp!, jacobian, u, p, dtγ, t, jac_resources)
    T_imp!.Wfact(jacobian, u, p, dtγ, t)
    return nothing
end

"""
    sync_jacobian_update!(token)

Wait for the asynchronous Jacobian update identified by `token` to complete.
No-op when `token === nothing` (synchronous backend).

There is no catch-all method: any backend whose [`async_Wfact!`](@ref)
returns a non-`nothing` token must extend `sync_jacobian_update!` for that
token type. Failing to do so raises `MethodError` at the next sync point —
this is intentional, since silently no-op'ing the wait would race the
default stream against the still-in-flight Jacobian assembly.
"""
@inline sync_jacobian_update!(::Nothing) = nothing

"""
    val_to_int(::Val{N}) -> N

Unwrap a `Val{N}` to its static integer. Used by the IMEX step loops to read
back the stage count from a `Val`-wrapped tableau dimension without
introducing a runtime type instability.
"""
@inline val_to_int(::Val{N}) where {N} = N

"""
    overlap_step_dispatch!(integrator, cache, a_imp, v_s)

Run the algorithm's stages, overlapping Jacobian assembly with stage 1 when
`a_imp[1, 1] == 0` (ESDIRK).
"""
function overlap_step_dispatch!(integrator, cache, a_imp, v_s)
    (; u, p, t, dt, alg) = integrator
    (; T_imp!) = integrator.sol.prob.f
    (; newtons_method) = alg
    (; γ, newtons_method_cache, jac_resources) = cache

    if iszero(a_imp[1, 1])
        # ESDIRK: launch Jacobian assembly in parallel with stage 1.
        jac_token = async_update_jacobian!(
            T_imp!,
            newtons_method,
            newtons_method_cache,
            u,
            p,
            t,
            dt,
            γ,
            alg,
            jac_resources,
        )
        update_stage!(integrator, cache, Val(1))
        sync_jacobian_update!(jac_token)

        s = val_to_int(v_s)
        update_stage!(integrator, cache, ntuple(j -> Val(j + 1), Val(s - 1)))
    else
        # Implicit-first-stage: Jacobian must be ready before stage 1.
        maybe_update_jacobian!(
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
        update_stage!(integrator, cache, ntuple(j -> Val(j), v_s))
    end
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
