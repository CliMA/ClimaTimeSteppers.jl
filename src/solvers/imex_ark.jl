import NVTX

has_jac(T_imp!) =
    hasfield(typeof(T_imp!), :Wfact) &&
    hasfield(typeof(T_imp!), :jac_prototype) &&
    !isnothing(T_imp!.Wfact) &&
    !isnothing(T_imp!.jac_prototype)

"""
    IMEXARKCache{SCU, SCE, SCI, T, Γ, NMC, TAB}

Pre-allocated workspace for an unconstrained IMEX ARK timestep.

# Fields
- `U`: stage state (sparse container of length `s`).
- `T_lim`: limited explicit tendency at each stage.
- `T_exp`: explicit tendency at each stage.
- `T_imp`: implicit tendency at each stage.
- `temp`: scratch array for the implicit solve.
- `γ`: common SDIRK diagonal coefficient (or `nothing` for non-SDIRK).
- `newtons_method_cache`: cache for [`NewtonsMethod`](@ref).
- `tableau`: the [`IMEXTableau`](@ref) with optional eltype cast.
"""
struct IMEXARKCache{SCU, SCE, SCI, T, Γ, NMC, TAB}
    U::SCU
    T_lim::SCE
    T_exp::SCE
    T_imp::SCI
    temp::T
    γ::Γ
    newtons_method_cache::NMC
    tableau::TAB
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
    check_sdirk_compat(alg, γ)
    jac_prototype = has_jac(T_imp!) ? T_imp!.jac_prototype : nothing
    newtons_method_cache =
        isnothing(T_imp!) || isnothing(newtons_method) ? nothing :
        allocate_cache(newtons_method, u0, jac_prototype)

    # Cast tableau coefficients to `eltype(u0)` if requested via
    # `cast_tableau_to_state_eltype`; otherwise preserve the tableau's eltype.
    opt_tb =
        alg.options.cast_tableau_to_state_eltype ? downcast_tableau(eltype(u0), tableau) :
        tableau

    return IMEXARKCache(
        U, T_lim, T_exp, T_imp, temp, γ, newtons_method_cache, opt_tb,
    )
end

# generic fallback
function step_u!(integrator, cache::IMEXARKCache)
    (; u, p, t, dt) = integrator
    (; f) = integrator.sol.prob
    (; cache!, T_imp!, lim!, dss!) = f
    (; b_exp, a_imp, b_imp) = cache.tableau
    (; T_lim, T_exp, T_imp, temp) = cache
    v_s = get_val_S(b_imp)

    step_stages!(integrator, cache, v_s)

    t_final = t + dt

    inc_exp = has_T_exp(f) ? fused_raw_increment(dt, b_exp, T_exp, v_s) : NullBroadcasted()
    inc_imp =
        isnothing(T_imp!) ? NullBroadcasted() : fused_raw_increment(dt, b_imp, T_imp, v_s)

    if has_T_lim(f)
        assign_fused_increment!(temp, u, dt, b_exp, T_lim, v_s)
        lim!(temp, p, t_final, u)
        assign_with_increments!(u, temp, inc_exp, inc_imp)
    else
        assign_with_increments!(u, u, inc_exp, inc_imp)
    end

    dss!(u, p, t_final)
    cache!(u, p, t_final)

    return u
end


@inline update_stage!(integrator, cache, ::Tuple{}) = nothing
@inline update_stage!(integrator, cache, is::Tuple{Any}) =
    update_stage!(integrator, cache, first(is))
@inline function update_stage!(integrator, cache, is::Tuple)
    update_stage!(integrator, cache, first(is))
    update_stage!(integrator, cache, Base.tail(is))
end
@inline function update_stage!(integrator, cache::IMEXARKCache, v_i::Val{i}) where {i}
    (; u, p, t, dt, alg) = integrator
    (; f) = integrator.sol.prob
    (; a_exp, b_exp, a_imp, b_imp, c_exp, c_imp) = cache.tableau
    (; U, T_lim, T_exp, T_imp, temp) = cache

    t_exp = t + dt * c_exp[i]
    t_imp = t + dt * c_imp[i]
    dtγ = float(dt) * a_imp[i, i]

    compute_stage_value!(U, u, dt, a_exp, a_imp, T_lim, T_exp, T_imp, f, p, t_exp, v_i)
    solve_stage_implicit!(U, temp, p, t_exp, t_imp, dtγ, i, f, alg, cache)
    evaluate_stage_tendencies!(
        T_exp, T_lim, T_imp, U, temp,
        p, t_exp, t_imp, dtγ, v_i, f, a_exp, b_exp, a_imp, b_imp,
    )

    return nothing
end

"""
    compute_stage_value!(U, u, dt, a_exp, a_imp, T_lim, T_exp, T_imp, f, p, t_exp, v_i::Val{i})

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
    v_i::Val{i},
) where {i}
    inc_exp = has_T_exp(f) ? fused_raw_increment(dt, a_exp, T_exp, v_i) : NullBroadcasted()
    inc_imp =
        isnothing(f.T_imp!) ? NullBroadcasted() : fused_raw_increment(dt, a_imp, T_imp, v_i)

    if has_T_lim(f)
        assign_fused_increment!(U, u, dt, a_exp, T_lim, v_i)
        i ≠ 1 && f.lim!(U, p, t_exp, u)
        assign_with_increments!(U, U, inc_exp, inc_imp)
    else
        assign_with_increments!(U, u, inc_exp, inc_imp)
    end
end

"""
    solve_stage_implicit!(U, temp, p, t_exp, t_imp, dtγ, i, f, alg, cache)

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

    solve_implicit_equation!(
        U, temp, p, t_imp, dtγ,
        T_imp!, newtons_method, newtons_method_cache,
        cache_imp!, dss!, cache!,
    )
end

"""
    evaluate_stage_tendencies!(T_exp, T_lim, T_imp, U, temp,
                               p, t_exp, t_imp, dtγ, v_i, f,
                               a_exp, b_exp, a_imp, b_imp)

Evaluate explicit and implicit tendencies at the current stage value for use
in later stages and the final update.
"""
@inline function evaluate_stage_tendencies!(
    T_exp, T_lim, T_imp, U, temp,
    p, t_exp, t_imp, dtγ, v_i::Val{i}, f, a_exp, b_exp, a_imp, b_imp,
) where {i}
    has_exp_terms = !zero_column(typeof(a_exp), v_i) || !zero_coeff(typeof(b_exp), i)
    if has_exp_terms
        isnothing(f.T_exp_T_lim!) || f.T_exp_T_lim!(T_exp[i], T_lim[i], U, p, t_exp)
    end

    # Implicit tendencies
    has_imp_terms = !zero_column(typeof(a_imp), v_i) || !zero_coeff(typeof(b_imp), i)
    if has_imp_terms
        if iszero(dtγ)
            # γ == 0: T_imp is treated explicitly
            isnothing(f.T_imp!) || f.T_imp!(T_imp[i], U, p, t_imp)
        else
            # γ ≠ 0: T_imp satisfies the implicit equation
            isnothing(f.T_imp!) || @. T_imp[i] = (U - temp) / dtγ
        end
    end
end

"""
    solve_implicit_equation!(U, U₀, p, t_imp, dtγ, T!, newtons_method, newtons_method_cache, cache_imp!, dss!, cache!)

Solve the implicit stage equation ``U = U₀ + Δtγ\\, T(U)`` via Newton's method.

Called from [`update_stage!`](@ref) for each implicit stage.
"""
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
