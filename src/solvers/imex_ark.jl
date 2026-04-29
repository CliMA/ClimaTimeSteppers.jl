import NVTX

has_jac(T_imp!) =
    hasfield(typeof(T_imp!), :Wfact) &&
    hasfield(typeof(T_imp!), :jac_prototype) &&
    !isnothing(T_imp!.Wfact) &&
    !isnothing(T_imp!.jac_prototype)

sdirk_error(name) = error("$(isnothing(name) ? "The given IMEXTableau" : name) \
                           has implicit stages with distinct coefficients (it \
                           is not SDIRK), and an update is required whenever a \
                           stage has a different coefficient from the previous \
                           stage. Do not update on the NewTimeStep signal when \
                           using $(isnothing(name) ? "this tableau" : name).")

"""
    maybe_update_jacobian!(T_imp!, newtons_method, cache, u, p, t, dt, γ, alg)

Update the Jacobian at the start of a timestep if the update signal fires.
Shared by IMEX ARK and IMEX SSPRK solvers.
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
    isnothing(T_imp!) && return
    isnothing(newtons_method) && return
    (; update_j) = newtons_method
    jacobian = newtons_method_cache.j
    isnothing(jacobian) && return
    if needs_update!(update_j, NewTimeStep(t))
        if γ isa Nothing
            sdirk_error(alg.name)
        else
            T_imp!.Wfact(jacobian, u, p, dt * γ, t)
        end
    end
    return nothing
end

struct IMEXARKCache{SCU, SCE, SCI, T, Γ, NMC}
    U::SCU     # sparse container of length s
    T_lim::SCE # sparse container of length s
    T_exp::SCE # sparse container of length s
    T_imp::SCI # sparse container of length s
    temp::T
    γ::Γ
    newtons_method_cache::NMC
end

function init_cache(prob, alg::IMEXAlgorithm{Unconstrained}; kwargs...)
    (; u0, f) = prob
    (; T_imp!) = f
    (; tableau, newtons_method) = alg
    (; a_exp, b_exp, a_imp, b_imp) = tableau
    s = length(b_exp)
    inds = ntuple(i -> i, s)
    inds_T_exp = filter(i -> !all(iszero, a_exp[:, i]) || !iszero(b_exp[i]), inds)
    inds_T_imp = filter(i -> !all(iszero, a_imp[:, i]) || !iszero(b_imp[i]), inds)
    U = zero(u0)
    T_lim = SparseContainer(map(i -> zero(u0), collect(1:length(inds_T_exp))), inds_T_exp)
    T_exp = SparseContainer(map(i -> zero(u0), collect(1:length(inds_T_exp))), inds_T_exp)
    T_imp = SparseContainer(map(i -> zero(u0), collect(1:length(inds_T_imp))), inds_T_imp)
    temp = zero(u0)
    γs = unique(filter(!iszero, diag(a_imp)))
    γ = length(γs) == 1 ? γs[1] : nothing # TODO: This could just be a constant.
    jac_prototype = has_jac(T_imp!) ? T_imp!.jac_prototype : nothing
    newtons_method_cache =
        isnothing(T_imp!) || isnothing(newtons_method) ? nothing :
        allocate_cache(newtons_method, u0, jac_prototype)
    return IMEXARKCache(U, T_lim, T_exp, T_imp, temp, γ, newtons_method_cache)
end

# generic fallback
function step_u!(integrator, cache::IMEXARKCache)
    (; u, p, t, dt, alg) = integrator
    (; f) = integrator.sol.prob
    (; cache!, cache_imp!) = f
    (; T_exp_T_lim!, T_imp!, lim!, dss!) = f
    (; tableau, newtons_method) = alg
    (; a_exp, b_exp, a_imp, b_imp, c_exp, c_imp) = tableau
    (; U, T_lim, T_exp, T_imp, temp, γ, newtons_method_cache) = cache
    s = length(b_exp)

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

    update_stage!(integrator, cache, ntuple(i -> i, Val(s)))

    t_final = t + dt

    if has_T_lim(f) # Update based on limited tendencies from previous stages
        assign_fused_increment!(temp, u, dt, b_exp, T_lim, Val(s))
        lim!(temp, p, t_final, u)
        @. u = temp
    end

    # Update based on tendencies from previous stages
    has_T_exp(f) && fused_increment!(u, dt, b_exp, T_exp, Val(s))
    isnothing(T_imp!) || fused_increment!(u, dt, b_imp, T_imp, Val(s))

    dss!(u, p, t_final)
    cache!(u, p, t_final)

    return u
end


@inline update_stage!(integrator, cache, ::Tuple{}) = nothing
@inline update_stage!(integrator, cache, is::Tuple{Int}) =
    update_stage!(integrator, cache, first(is))
@inline function update_stage!(integrator, cache, is::Tuple)
    update_stage!(integrator, cache, first(is))
    update_stage!(integrator, cache, Base.tail(is))
end
@inline function update_stage!(integrator, cache::IMEXARKCache, i::Int)
    (; u, p, t, dt, alg) = integrator
    (; f) = integrator.sol.prob
    (; tableau) = alg
    (; a_exp, b_exp, a_imp, b_imp, c_exp, c_imp) = tableau
    (; U, T_lim, T_exp, T_imp, temp) = cache

    t_exp = t + dt * c_exp[i]
    t_imp = t + dt * c_imp[i]
    dtγ = float(dt) * a_imp[i, i]

    # 1. Compute stage value from previous tendencies
    compute_stage_value!(U, u, dt, a_exp, a_imp, T_lim, T_exp, T_imp, f, p, t_exp, i)

    # 2. Apply DSS (skip stage 1) + implicit solve
    solve_stage_implicit!(U, temp, p, t_exp, t_imp, dtγ, i, f, alg, cache)

    # 3. Evaluate tendencies for later stages and final update
    evaluate_stage_tendencies!(
        T_exp, T_lim, T_imp, U, temp,
        p, t_exp, t_imp, dtγ, i, f, a_exp, b_exp, a_imp, b_imp,
    )

    return nothing
end

"""
    compute_stage_value!(U, u, dt, a_exp, a_imp, T_lim, T_exp, T_imp, f, p, t_exp, i)

Compute the stage value `U` by accumulating the linear combination dictated by the 
Butcher tableau from previous explicit (limited + unlimited) and implicit tendencies.
"""
@inline function compute_stage_value!(
    U,
    u,
    dt,
    a_exp,
    a_imp,
    T_lim,
    T_exp,
    T_imp,
    f,
    p,
    t_exp,
    i,
)
    if has_T_lim(f)
        assign_fused_increment!(U, u, dt, a_exp, T_lim, Val(i))
        i ≠ 1 && f.lim!(U, p, t_exp, u)
    else
        @. U = u
    end
    has_T_exp(f) && fused_increment!(U, dt, a_exp, T_exp, Val(i))
    isnothing(f.T_imp!) || fused_increment!(U, dt, a_imp, T_imp, Val(i))
end

"""
    solve_stage_implicit!(U, temp, temp_sub, p, t_exp, t_imp, dtγ, i, f, alg, cache)

Apply DSS to the stage value and run the implicit solver.
Skips DSS on stage 1 (already applied at end of previous timestep).
Skips implicit solve when γ == 0.
"""
@inline function solve_stage_implicit!(
    U,
    temp,
    p,
    t_exp,
    t_imp,
    dtγ,
    i,
    f,
    alg,
    cache,
)
    (; T_imp!, dss!, cache!, cache_imp!, initialize_imp!) = f
    (; newtons_method_cache) = cache
    (; newtons_method) = alg

    i ≠ 1 && dss!(U, p, t_exp)

    if isnothing(T_imp!) || iszero(dtγ)
        i ≠ 1 && cache!(U, p, t_exp)
        return
    end

    @assert !isnothing(newtons_method)
    @. temp = U

    if !isnothing(initialize_imp!)
        initialize_imp!(U, p, dtγ)
        dss!(U, p, t_exp)
        cache_imp!(U, p, t_imp)
    else
        i ≠ 1 && cache_imp!(U, p, t_imp)
    end

    # Main implicit solve
    solve_implicit_equation!(
        U, temp, p, t_imp, dtγ,
        T_imp!, newtons_method, newtons_method_cache,
        cache_imp!, dss!, cache!,
    )
end

"""
    evaluate_stage_tendencies!(T_exp, T_lim, T_imp, U, temp, temp_sub, ...)

Evaluate explicit and implicit tendencies at the current stage value for use
in later stages and the final update.
"""
@inline function evaluate_stage_tendencies!(
    T_exp, T_lim, T_imp, U, temp,
    p, t_exp, t_imp, dtγ, i, f, a_exp, b_exp, a_imp, b_imp,
)
    # Explicit tendencies (needed for later stages or final update)
    if !all(iszero, a_exp[:, i]) || !iszero(b_exp[i])
        isnothing(f.T_exp_T_lim!) || f.T_exp_T_lim!(T_exp[i], T_lim[i], U, p, t_exp)
    end

    # Implicit tendencies
    if !all(iszero, a_imp[:, i]) || !iszero(b_imp[i])
        if iszero(dtγ)
            # γ == 0: T_imp is treated explicitly
            isnothing(f.T_imp!) || f.T_imp!(T_imp[i], U, p, t_imp)
        else
            # γ ≠ 0: T_imp satisfies the implicit equation
            isnothing(f.T_imp!) || @. T_imp[i] = (U - temp) / dtγ
        end
    end
end

@inline function solve_implicit_equation!(
    U,
    U₀,
    p,
    t_imp,
    dtγ,
    T!,
    newtons_method,
    newtons_method_cache,
    cache_imp!,
    dss!,
    cache!,
)
    implicit_equation_residual! =
        (residual, U′) -> begin
            T!(residual, U′, p, t_imp)
            @. residual = U₀ + dtγ * residual - U′
        end
    solve_newton!(
        newtons_method,
        newtons_method_cache,
        U,
        implicit_equation_residual!,
        (jacobian, U′) -> T!.Wfact(jacobian, U′, p, dtγ, t_imp),
        U′ -> cache_imp!(U′, p, t_imp),
    )
    dss!(U, p, t_imp)
    cache!(U, p, t_imp)
    return nothing
end
