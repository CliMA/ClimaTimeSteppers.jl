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

"""
    val_to_int(::Val{N})
    
Extracts the statically-known integer from a Val type wrapper.
"""
@inline val_to_int(::Val{N}) where {N} = N

"""
    overlap_step_dispatch!(integrator, cache, a_imp, v_s)

Orchestrates the stage executions and Jacobian updates, possibly overlapping them
if the Butcher tableau permits (i.e. if a_imp[1,1] is zero).
"""
function overlap_step_dispatch!(integrator, cache, a_imp, v_s)
    (; u, p, t, dt, alg) = integrator
    (; f) = integrator.sol.prob
    (; T_imp!) = f
    (; newtons_method) = alg
    (; γ, newtons_method_cache) = cache

    if iszero(a_imp[1, 1])
        # Explicit-first-stage (ESDIRK): Overlap Jacobian build with Stage 1.
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
        )
        # Execute Stage 1 (which has no dependence on the fresh Jacobian)
        update_stage!(integrator, cache, Val(1))

        # Synchronize: Ensure the new Jacobian is completely assembled before starting Stage 2
        sync_jacobian_update!(jac_token)

        # Statically iterate through the remaining stages
        s_val = val_to_int(v_s)
        update_stage!(integrator, cache, ntuple(j -> Val(j + 1), Val(s_val - 1)))
    else
        # Implicit-first-stage: Fallback to strictly sequential execution
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
