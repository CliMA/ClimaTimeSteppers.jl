# Utility interface for asynchronous execution orchestration.
# The functions fallback to basic synchronous execution for basic backends.
# Native GPU acceleration via multiple streams is injected via weak-dependency extensions.

"""
    async_update_jacobian!(T_imp!, newtons_method, cache, u, p, t, dt, γ, alg)

Centralized logic deciding if an update is due. If so, delegates the computationally
heavy work to `async_Wfact!`.
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
)
    isnothing(T_imp!) && return nothing
    isnothing(newtons_method) && return nothing
    (; update_j) = newtons_method
    jacobian = newtons_method_cache.j
    isnothing(jacobian) && return nothing

    if needs_update!(update_j, NewTimeStep(t))
        if γ isa Nothing
            sdirk_error(alg.name)
        else
            # Logic confirms update is mandatory. Invoke low-level launch.
            return async_Wfact!(T_imp!, jacobian, u, p, dt * γ, t)
        end
    end
    return nothing
end

"""
    async_Wfact!(T_imp!, jacobian, u, p, dtγ, t)

Launches the Wfact matrix computation. Specialized by backend extensions (e.g. CUDA)
to fork onto different device streams.
"""
function async_Wfact!(T_imp!, jacobian, u, p, dtγ, t)
    # Sequential CPU baseline
    T_imp!.Wfact(jacobian, u, p, dtγ, t)
    return nothing
end

"""
    sync_jacobian_update!(token)

Wait for completion of background jacobian construction task identified by token.
Defaults to no-op for synchronous fallback paths.
"""
@inline function sync_jacobian_update!(token)
    return nothing
end
