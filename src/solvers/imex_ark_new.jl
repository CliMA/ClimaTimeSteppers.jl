has_jac(T_imp!) =
    hasfield(typeof(T_imp!), :Wfact) &&
    hasfield(typeof(T_imp!), :jac_prototype) &&
    !isnothing(T_imp!.Wfact) &&
    !isnothing(T_imp!.jac_prototype)

imp_error(name) = error("$(isnothing(name) ? "The given IMEXTableau" : name) \
                         has implicit stages that require a nonlinear solver, \
                         so NewtonsMethod must be specified alongside T_imp!.")

sdirk_error(name) = error("$(isnothing(name) ? "The given IMEXTableau" : name) \
                           has implicit stages with distinct coefficients (it \
                           is not SDIRK), and an update is required whenever a \
                           stage has a different coefficient from the previous \
                           stage. Do not update on the NewTimeStep signal when \
                           using $(isnothing(name) ? "this tableau" : name).")

struct IMEXARKCache{T, N}
    timestepper_cache::T
    newtons_method_cache::N
end

function init_cache(prob::DiffEqBase.AbstractODEProblem, alg::IMEXAlgorithm{Unconstrained}; kwargs...)
    (; u0) = prob
    (; T_lim!, T_exp!, T_exp_T_lim!, T_imp!) = prob.f
    (; name, newtons_method) = alg
    (; a_exp, b_exp, a_imp, b_imp) = alg.tableau

    no_T_lim = isnothing(T_lim!) && isnothing(T_exp_T_lim!)
    no_T_exp = isnothing(T_exp!) && isnothing(T_exp_T_lim!)
    no_T_imp = isnothing(T_imp!)

    s = size(a_imp, 1) # number of internal stages

    # Extend the coefficient matrices a_exp and a_imp by interpreting the final
    # state on each step as stage s + 1.
    A_exp = vcat(a_exp, b_exp')
    A_imp = vcat(a_imp, b_imp')

    z_stages = findall(iszero, diag(A_imp)) # stages without implicit solves
    nz_stages = findall(!iszero, diag(A_imp)) # stages with implicit solves
    stages_needing_T_exp = findall(col -> any(!iszero, col), eachcol(A_exp))
    z_stages_needing_T_imp = findall(col -> any(!iszero, col), eachcol(A_imp[:, z_stages]))
    # All nz stages are computed using ΔU_imp, rather than T_imp.

    Γ = A_imp[nz_stages, nz_stages] # "fully implicit" part of A_imp
    sdirk_γ = length(unique(diag(Γ))) == 1 ? diag(Γ)[1] : nothing

    temp_value1 = similar(u0)
    temp_value2 = similar(u0)
    T_lim_values_sparse = no_T_lim ? SparseTuple() : SparseTuple(_ -> similar(u0), stages_needing_T_exp)
    T_exp_values_sparse = no_T_exp ? SparseTuple() : SparseTuple(_ -> similar(u0), stages_needing_T_exp)
    T_imp_values_sparse = no_T_imp ? SparseTuple() : SparseTuple(_ -> similar(u0), z_stages_needing_T_imp)
    ΔU_imp_values_sparse = no_T_imp ? SparseTuple() : SparseTuple(_ -> similar(u0), nz_stages)

    ΔtT_lim_to_Δu_lim_tuples_sparse = no_T_lim ? SparseTuple() : sparse_matrix_rows(A_exp, 1:(s + 1), 1:s)
    ΔtT_exp_to_Δu_exp_tuples_sparse = no_T_exp ? SparseTuple() : sparse_matrix_rows(A_exp, 1:(s + 1), 1:s)
    ΔtT_imp_to_Δu_imp_tuples_sparse =
        no_T_imp ? SparseTuple() : sparse_matrix_rows(A_imp[:, z_stages], 1:(s + 1), z_stages)

    A_imp_lower_nz_component = A_imp[:, nz_stages]
    A_imp_lower_nz_component[nz_stages, nz_stages] .= Γ - Diagonal(Γ)
    prev_ΔU_imp_to_Δu_imp_tuples_sparse =
        no_T_imp ? SparseTuple() : sparse_matrix_rows(A_imp_lower_nz_component * inv(Γ), 1:(s + 1), nz_stages)

    # Convert all values that will be passed to unrolled_foreach in step_u! into
    # tuples of length s.
    T_lim_values = dense_tuple(T_lim_values_sparse, s, nothing)
    T_exp_values = dense_tuple(T_exp_values_sparse, s, nothing)
    T_imp_values = dense_tuple(T_imp_values_sparse, s, nothing)
    ΔU_imp_values = dense_tuple(ΔU_imp_values_sparse, s, nothing)
    ΔtT_lim_to_Δu_lim_tuples = dense_tuple(ΔtT_lim_to_Δu_lim_tuples_sparse, s, SparseTuple())
    ΔtT_exp_to_Δu_exp_tuples = dense_tuple(ΔtT_exp_to_Δu_exp_tuples_sparse, s, SparseTuple())
    ΔtT_imp_to_Δu_imp_tuples = dense_tuple(ΔtT_imp_to_Δu_imp_tuples_sparse, s, SparseTuple())
    prev_ΔU_imp_to_Δu_imp_tuples = dense_tuple(prev_ΔU_imp_to_Δu_imp_tuples_sparse, s, SparseTuple())

    timestepper_cache = (;
        sdirk_γ,
        temp_value1,
        temp_value2,
        T_lim_values_sparse,
        T_exp_values_sparse,
        T_imp_values_sparse,
        ΔU_imp_values_sparse,
        T_lim_values,
        T_exp_values,
        T_imp_values,
        ΔU_imp_values,
        ΔtT_lim_to_Δu_lim_tuples,
        ΔtT_exp_to_Δu_exp_tuples,
        ΔtT_imp_to_Δu_imp_tuples,
        prev_ΔU_imp_to_Δu_imp_tuples,
    )

    newtons_method_cache = if is_accessible(ΔU_imp_values_sparse)
        isnothing(newtons_method) && imp_error(name)
        j = has_jac(T_imp!) ? T_imp!.jac_prototype : nothing
        allocate_cache(newtons_method, u0, j)
    else
        nothing
    end

    return IMEXARKCache(timestepper_cache, newtons_method_cache)
end

function step_u!(integrator, cache::IMEXARKCache)
    (; u, p, t, alg) = integrator
    (; T_lim!, T_exp!, T_exp_T_lim!, T_imp!) = integrator.sol.prob.f
    (; lim!, dss!, post_explicit!, post_implicit!) = integrator.sol.prob.f
    (; name, newtons_method) = alg
    (; a_imp, c_exp, c_imp) = alg.tableau
    (; newtons_method_cache) = cache
    (;
        sdirk_γ,
        temp_value1,
        temp_value2,
        T_lim_values_sparse,
        T_exp_values_sparse,
        T_imp_values_sparse,
        ΔU_imp_values_sparse,
        T_lim_values,
        T_exp_values,
        T_imp_values,
        ΔU_imp_values,
        ΔtT_lim_to_Δu_lim_tuples,
        ΔtT_exp_to_Δu_exp_tuples,
        ΔtT_imp_to_Δu_imp_tuples,
        prev_ΔU_imp_to_Δu_imp_tuples,
    ) = cache.timestepper_cache

    Δt = integrator.dt
    s = size(a_imp, 1)

    if !isnothing(newtons_method_cache)
        (; update_j) = newtons_method
        (; j) = newtons_method_cache
        if !isnothing(j) && needs_update!(update_j, NewTimeStep(t))
            isnothing(sdirk_γ) && sdirk_error(name)
            T_imp!.Wfact(j, u, p, Δt * sdirk_γ, t)
        end
    end

    unrolled_foreach(
        ntuple(identity, s),
        T_lim_values,
        T_exp_values,
        T_imp_values,
        ΔU_imp_values,
        ΔtT_lim_to_Δu_lim_tuples,
        ΔtT_exp_to_Δu_exp_tuples,
        ΔtT_imp_to_Δu_imp_tuples,
        prev_ΔU_imp_to_Δu_imp_tuples,
    ) do (
        stage,
        T_lim,
        T_exp,
        T_imp,
        ΔU_imp,
        ΔtT_lim_to_ΔU_lim,
        ΔtT_exp_to_ΔU_exp,
        ΔtT_imp_to_ΔU_imp,
        prev_ΔU_imp_to_ΔU_imp,
    )
        t_exp = t + Δt * c_exp[stage]
        t_imp = t + Δt * c_imp[stage]
        Δtγ = Δt * a_imp[stage, stage]

        ΔU_lim_over_Δt = sparse_broadcasted_dot(ΔtT_lim_to_ΔU_lim, T_lim_values_sparse)
        ΔU_exp_over_Δt = sparse_broadcasted_dot(ΔtT_exp_to_ΔU_exp, T_exp_values_sparse)
        ΔU_imp_from_T_imp_over_Δt = sparse_broadcasted_dot(ΔtT_imp_to_ΔU_imp, T_imp_values_sparse)
        ΔU_imp_from_prev_ΔU_imp = sparse_broadcasted_dot(prev_ΔU_imp_to_ΔU_imp, ΔU_imp_values_sparse)

        if is_accessible(ΔtT_lim_to_ΔU_lim)
            u_plus_ΔU_lim = temp_value1
            @. u_plus_ΔU_lim = u + Δt * ΔU_lim_over_Δt
            lim!(u_plus_ΔU_lim, p, t_exp, u)
        else
            u_plus_ΔU_lim = u
        end

        if is_accessible(ΔtT_exp_to_ΔU_exp) || is_accessible(ΔtT_imp_to_ΔU_imp) || is_accessible(prev_ΔU_imp_to_ΔU_imp)
            U_before_solve = temp_value2
            @. U_before_solve =
                u_plus_ΔU_lim + Δt * (ΔU_exp_over_Δt + ΔU_imp_from_T_imp_over_Δt) + ΔU_imp_from_prev_ΔU_imp
        else
            U_before_solve = u_plus_ΔU_lim
        end

        is_not_u_before_solve =
            is_accessible(ΔtT_lim_to_ΔU_lim) ||
            is_accessible(ΔtT_exp_to_ΔU_exp) ||
            is_accessible(ΔtT_imp_to_ΔU_imp) ||
            is_accessible(prev_ΔU_imp_to_ΔU_imp)

        # TODO: Rename post_explicit! to pre_newton_iteration!, and rename
        # post_implicit! to post_stage!. Make pre_newton_iteration! only set
        # precomputed quantities needed by T_imp! and T_imp!.Wfact. Keep
        # post_stage! as it is now, so that it sets all precomputed quantities.

        if !isnothing(ΔU_imp)
            is_not_u_before_solve && post_explicit!(U_before_solve, p, t_imp)

            U = ΔU_imp # Use ΔU_imp as additional temporary storage.
            @. U = U_before_solve

            # Solve U ≈ U_before_solve + Δtγ * T_imp(U, p, t_imp) for U.
            solve_newton!(
                newtons_method,
                newtons_method_cache,
                U,
                (residual, U) -> begin
                    T_imp!(residual, U, p, t_imp)
                    @. residual = U_before_solve - U + Δtγ * residual
                end,
                (j, U) -> T_imp!.Wfact(j, U, p, Δtγ, t_imp), # j = ∂residual/∂U
                U -> post_explicit!(U, p, t_imp),
                U -> post_implicit!(U, p, t_imp),
            )
        else
            U = U_before_solve # There is no solve on this stage.
            is_not_u_before_solve && post_implicit!(U, p, t_imp)
        end

        if !isnothing(T_lim) || !isnothing(T_exp)
            if !isnothing(T_exp_T_lim!)
                T_exp_T_lim!(T_exp, T_lim, U, p, t_exp)
                if stage != s
                    # TODO: Fuse these two DSS calls into one.
                    dss!(T_lim, p, t_exp)
                    dss!(T_exp, p, t_exp)
                end
            end
            # TODO: Drop support for specifying T_lim! separately from T_exp!.
            if !isnothing(T_lim!)
                T_lim!(T_lim, U, p, t_exp)
                stage != s && dss!(T_lim, p, t_exp)
            end
            if !isnothing(T_exp!)
                T_exp!(T_exp, U, p, t_exp)
                stage != s && dss!(T_exp, p, t_exp)
            end
        end
        if !isnothing(T_imp)
            T_imp!(T_imp, U, p, t_imp)
        end
        if !isnothing(ΔU_imp)
            @. ΔU_imp = U - u_plus_ΔU_lim - Δt * ΔU_exp_over_Δt
            #         = U - U_before_solve + Δt * ΔU_imp_from_T_imp_over_Δt +
            #           ΔU_imp_from_prev_ΔU_imp
        end
    end

    t_final = t + Δt
    ΔtT_lim_to_final_Δu_lim = ΔtT_lim_to_Δu_lim_tuples[s + 1]
    ΔtT_exp_to_final_Δu_exp = ΔtT_exp_to_Δu_exp_tuples[s + 1]
    ΔtT_imp_to_final_Δu_imp = ΔtT_imp_to_Δu_imp_tuples[s + 1]
    ΔU_imp_to_final_Δu_imp = prev_ΔU_imp_to_Δu_imp_tuples[s + 1]
    final_Δu_lim_over_Δt = sparse_broadcasted_dot(ΔtT_lim_to_final_Δu_lim, T_lim_values_sparse)
    final_Δu_exp_over_Δt = sparse_broadcasted_dot(ΔtT_exp_to_final_Δu_exp, T_exp_values_sparse)
    final_Δu_imp_from_T_imp_over_Δt = sparse_broadcasted_dot(ΔtT_imp_to_final_Δu_imp, T_imp_values_sparse)
    final_Δu_imp_from_ΔU_imp = sparse_broadcasted_dot(ΔU_imp_to_final_Δu_imp, ΔU_imp_values_sparse)

    if is_accessible(ΔtT_lim_to_final_Δu_lim)
        final_u_plus_Δu_lim = temp_value
        @. final_u_plus_Δu_lim = u + Δt * final_Δu_lim_over_Δt
        lim!(final_u_plus_Δu_lim, p, t_final, u)
    else
        final_u_plus_Δu_lim = u
    end

    @. u =
        final_u_plus_Δu_lim + Δt * (final_Δu_exp_over_Δt + final_Δu_imp_from_T_imp_over_Δt) + final_Δu_imp_from_ΔU_imp
    dss!(u, p, t_final)
    post_implicit!(u, p, t_final)

    return u
end
