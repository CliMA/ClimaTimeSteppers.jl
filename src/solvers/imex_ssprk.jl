"""
    IMEXSSPRKCache{U, SCI, B, Γ, NMC, TAB}

Pre-allocated workspace for an SSP-constrained IMEX SSPRK timestep.

# Fields
- `U`: stage state.
- `U_exp`: explicit stage state.
- `U_lim`: stage state for the limiter update.
- `T_lim`: limited explicit tendency.
- `T_exp`: explicit tendency.
- `T_imp`: implicit tendency (sparse container of length `s`).
- `temp`: scratch array for the implicit solve.
- `β`: low-storage Shu-Osher coefficients of the explicit SSPRK tableau.
- `γ`: common SDIRK diagonal coefficient (or `nothing` for non-SDIRK).
- `newtons_method_cache`: cache for [`NewtonsMethod`](@ref).
- `tableau`: the [`IMEXTableau`](@ref) with optional eltype cast.
"""
struct IMEXSSPRKCache{U, SCI, B, Γ, NMC, TAB}
    U::U
    U_exp::U
    U_lim::U
    T_lim::U
    T_exp::U
    T_imp::SCI
    temp::U
    β::B
    γ::Γ
    newtons_method_cache::NMC
    tableau::TAB
end

function init_cache(prob, alg::IMEXAlgorithm{SSP}; kwargs...)
    (; u0, f) = prob
    (; T_imp!) = f
    (; tableau, newtons_method) = alg
    (; a_exp, b_exp, a_imp, b_imp) = tableau
    s = length(b_exp)
    inds = ntuple(i -> i, s)
    inds_T_imp = filter(i -> !all(iszero, a_imp[:, i]) || !iszero(b_imp[i]), inds)
    U = zero(u0)
    U_exp = zero(u0)
    T_lim = zero(u0)
    T_exp = zero(u0)
    U_lim = zero(u0)
    T_imp = SparseContainer(map(i -> zero(u0), collect(1:length(inds_T_imp))), inds_T_imp)
    temp = zero(u0)
    â_exp = SparseCoeffs(vcat(a_exp.coeffs, b_exp.coeffs'))
    β = SparseCoeffs(diag(â_exp, -1))
    for i in 1:length(β)
        if â_exp.coeffs[(i + 1):end, i] != cumprod(β.coeffs[i:end])
            error("The SSP IMEXAlgorithm currently only supports an \
                   IMEXTableau that specifies a \"low-storage\" IMEX SSPRK \
                   algorithm, where the canonical Shu-Osher representation of \
                   the i-th explicit stage for i > 1 must have the form U[i] = \
                   (1 - β[i-1]) * u + β[i-1] * (U[i-1] + dt * T_exp(U[i-1])). \
                   So, it must be possible to express vcat(a_exp, b_exp') as\n \
                   0                  0           0    …\n \
                   β[1]               0           0    …\n \
                   β[1] * β[2]        β[2]        0    …\n \
                   β[1] * β[2] * β[3] β[2] * β[3] β[3] …\n \
                   ⋮                  ⋮            ⋮    ⋱\n \
                   The given IMEXTableau does not satisfy this property.")
        end
    end
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

    return IMEXSSPRKCache(
        U,
        U_exp,
        U_lim,
        T_lim,
        T_exp,
        T_imp,
        temp,
        β,
        γ,
        newtons_method_cache,
        opt_tb,
    )
end

function step_u!(integrator, cache::IMEXSSPRKCache)
    (; u, p, t, dt) = integrator
    (; f) = integrator.sol.prob
    (; cache!, T_imp!, lim!, dss!, constrain_state!) = f
    (; b_imp) = cache.tableau
    (; U_exp, U_lim, T_lim, T_exp, T_imp, β) = cache
    v_s = get_val_S(b_imp)

    step_stages!(integrator, cache, v_s)

    t_final = t + dt

    inc_imp_final = isnothing(T_imp!) ? 0 : fused_raw_increment(dt, b_imp, T_imp, v_s)

    s = val_to_int(v_s)
    if !iszero(β[s])
        if has_T_lim(f)
            @. U_lim = U_exp + dt * T_lim
            lim!(U_lim, p, t_final, U_exp)
            @. U_exp = U_lim
        end
        if has_T_exp(f)
            @. U_exp += dt * T_exp
        end
        if inc_imp_final !== 0
            @. u = (1 - β[s]) * u + β[s] * U_exp + inc_imp_final
        else
            @. u = (1 - β[s]) * u + β[s] * U_exp
        end
    else
        if inc_imp_final !== 0
            @. u += inc_imp_final
        end
    end

    # End of step: fire `EndOfStepSignal` unconditionally. Because
    # `EndOfStep <: EndOfStage <: WithDSS`, all three handler families see
    # this single signal.
    dss!(u, p, t_final)
    needs_update!(f.update_constrain_state, EndOfStepSignal()) &&
        constrain_state!(u, p, t_final)
    needs_update!(f.update_cache, EndOfStepSignal()) && cache!(u, p, t_final)

    return u
end

@inline function update_stage!(integrator, cache::IMEXSSPRKCache, v_i::Val{i}) where {i}
    (; u, p, t, dt, alg) = integrator
    (; f) = integrator.sol.prob
    (; T_imp!, lim!) = f
    (; a_imp, b_imp, c_exp, c_imp) = cache.tableau
    (; U, U_lim, U_exp, T_lim, T_exp, T_imp, temp, β) = cache

    t_exp = t + dt * c_exp[i]
    t_imp = t + dt * c_imp[i]
    dtγ = float(dt) * a_imp[i, i]

    if i == 1
        @. U_exp = u
    elseif !iszero(β[i - 1])
        if has_T_lim(f)
            @. U_lim = U_exp + dt * T_lim
            lim!(U_lim, p, t_exp, U_exp)
            @. U_exp = U_lim
        end
        if has_T_exp(f)
            @. U_exp += dt * T_exp
        end
        @. U_exp = (1 - β[i - 1]) * u + β[i - 1] * U_exp
    end

    inc_imp = isnothing(T_imp!) ? 0 : fused_raw_increment(dt, a_imp, T_imp, v_i)
    @. U = U_exp + inc_imp

    # Apply DSS, set up and run the implicit solve (honoring `initialize_imp!`),
    # and update the cache/constraints. Shared with the IMEX-ARK path so the two
    # stay in sync. SSPRK is never FSAL (its Shu-Osher assembly does not satisfy
    # `u ≡ U_s`), so the shared helper's last-stage skip never fires here.
    solve_stage_implicit!(U, temp, p, t_exp, t_imp, dtγ, i, f, alg, cache)

    if !iszero(β[i])
        isnothing(f.T_exp_T_lim!) || f.T_exp_T_lim!(T_exp, T_lim, U, p, t_exp)
    end
    if !zero_column(typeof(a_imp), v_i) || !zero_coeff(typeof(b_imp), i)
        if iszero(a_imp[i, i])
            # When γ == 0, T_imp[i] is treated explicitly.
            isnothing(T_imp!) || T_imp!(T_imp[i], U, p, t_imp)
        else
            # When γ != 0, T_imp[i] is treated implicitly, so it must satisfy
            # the implicit equation. To ensure that T_imp[i] only includes the
            # effect of applying DSS to T_imp!, U and temp must be DSSed.
            isnothing(T_imp!) || @. T_imp[i] = (U - temp) / dtγ
        end
    end
    return nothing
end
