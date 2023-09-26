function step_u!(integrator, cache::IMEXSSPRKCache, ::SSP333)
    (; u, p, t, dt, sol, alg) = integrator
    (; f) = sol.prob
    (; T_lim!, T_exp!, T_imp!, lim!, dss!) = f
    (; post_explicit!, post_implicit!) = f
    (; tableau, newtons_method) = alg
    (; a_imp, b_imp, c_exp, c_imp) = tableau
    (; U, U_lim, U_exp, T_lim, T_exp, T_imp, temp, β, γ, newtons_method_cache) = cache
    s = length(b_imp)

    if !isnothing(T_imp!) && !isnothing(newtons_method)
        (; update_j) = newtons_method
        jacobian = newtons_method_cache.j
        if (!isnothing(jacobian)) && needs_update!(update_j, NewTimeStep(t))
            if γ isa Nothing
                sdirk_error(name)
            else
                T_imp!.Wfact(jacobian, u, p, dt * γ, t)
            end
        end
    end

    @. U = u

    s = 3
    i::Int = 1
    t_exp = t
    t_imp = t

    @. U_exp = u
    dss!(U_exp, p, t_exp) # unnecessary
    @. U = U_exp

    # If its coefficient is 0, T_imp[i] is effectively being treated explicitly.
    T_imp!(T_imp[i], U, p, t_imp)
    T_lim!(T_lim, U, p, t_exp)
    T_exp!(T_exp, U, p, t_exp)

    i = 2
    t_exp = t + dt
    t_imp = t + dt

    @. U_lim = U_exp + dt * T_lim
    lim!(U_lim, p, t_exp, U_exp)
    @. U_exp = U_lim
    @. U_exp += dt * T_exp
    @. U_exp = (1 - β[i - 1]) * u + β[i - 1] * U_exp

    dss!(U_exp, p, t_exp)
    @. U = U_exp
    @. U += dt * a_imp[i, 1] * T_imp[1]
    post_explicit!(U, p, t_exp)

    @assert !isnothing(newtons_method)
    @. temp = U
    # TODO: can/should we remove these closures?
    implicit_equation_residual! = (residual, Ui) -> begin
        T_imp!(residual, Ui, p, t_imp)
        @. residual = temp + dt * a_imp[i, i] * residual - Ui
    end
    implicit_equation_jacobian! = (jacobian, Ui) -> T_imp!.Wfact(jacobian, Ui, p, dt * a_imp[i, i], t_imp)
    call_post_implicit! = Ui -> begin
        post_implicit!(Ui, p, t_imp)
    end
    solve_newton!(
        newtons_method,
        newtons_method_cache,
        U,
        implicit_equation_residual!,
        implicit_equation_jacobian!,
        call_post_implicit!,
    )

    # We do not need to DSS U again because the implicit solve should
    # give the same results for redundant columns (as long as the implicit
    # tendency only acts in the vertical direction).

    # If its coefficient is 0, T_imp[i] is effectively being
    # treated explicitly.
    T_imp!(T_imp[i], U, p, t_imp)
    T_lim!(T_lim, U, p, t_exp)
    T_exp!(T_exp, U, p, t_exp)

    i = 3
    t_exp = t + dt * c_exp[i]
    t_imp = t + dt * c_imp[i]

    @. U_lim = U_exp + dt * T_lim
    lim!(U_lim, p, t_exp, U_exp)
    @. U_exp = U_lim
    @. U_exp += dt * T_exp
    @. U_exp = (1 - β[i - 1]) * u + β[i - 1] * U_exp

    dss!(U_exp, p, t_exp)

    @. U = U_exp
    @. U += dt * a_imp[i, 1] * T_imp[1] + dt * a_imp[i, 2] * T_imp[2]
    post_explicit!(U, p, t_exp) # is t_exp correct here?

    @assert !isnothing(newtons_method)
    @. temp = U
    # TODO: can/should we remove these closures?
    implicit_equation_residual! = (residual, Ui) -> begin
        T_imp!(residual, Ui, p, t_imp)
        @. residual = temp + dt * a_imp[i, i] * residual - Ui
    end
    implicit_equation_jacobian! = (jacobian, Ui) -> T_imp!.Wfact(jacobian, Ui, p, dt * a_imp[i, i], t_imp)
    call_post_implicit! = Ui -> begin
        post_implicit!(Ui, p, t_imp)
    end
    solve_newton!(
        newtons_method,
        newtons_method_cache,
        U,
        implicit_equation_residual!,
        implicit_equation_jacobian!,
        call_post_implicit!,
    )

    # We do not need to DSS U again because the implicit solve should
    # give the same results for redundant columns (as long as the implicit
    # tendency only acts in the vertical direction).

    # If T_imp[i] is being treated implicitly, ensure that it
    # exactly satisfies the implicit equation.
    @. T_imp[i] = (U - temp) / (dt * a_imp[i, i])

    T_lim!(T_lim, U, p, t_exp)
    T_exp!(T_exp, U, p, t_exp)

    i = -1
    t_final = t + dt

    @. U_lim = U_exp + dt * T_lim
    lim!(U_lim, p, t_final, U_exp)
    @. U_exp = U_lim
    @. U_exp += dt * T_exp
    @. u = (1 - β[s]) * u + β[s] * U_exp
    @. u += dt * b_imp[1] * T_imp[1] + dt * b_imp[2] * T_imp[2] + dt * b_imp[3] * T_imp[3]

    dss!(u, p, t_final)
    post_explicit!(u, p, t_final)

    return u
end
