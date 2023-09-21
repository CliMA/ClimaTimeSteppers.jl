function step_u!(integrator, cache::IMEXARKCache, ::ARS343)
    (; u, p, t, dt, sol, alg) = integrator
    (; f) = sol.prob
    (; T_imp!, lim!, dss!) = f
    (; tableau, newtons_method) = alg
    (; a_exp, b_exp, a_imp, b_imp, c_exp, c_imp) = tableau
    (; U, T_lim, T_exp, T_imp, temp, γ, newtons_method_cache) = cache
    T_lim! = !isnothing(f.T_lim!) ? f.T_lim! : (args...) -> nothing
    T_exp! = !isnothing(f.T_exp!) ? f.T_exp! : (args...) -> nothing

    if !isnothing(newtons_method_cache)
        jacobian = newtons_method_cache.j
        if (!isnothing(jacobian)) && needs_update!(newtons_method.update_j, NewTimeStep(t))
            T_imp!.Wfact(jacobian, u, p, dt * γ, t)
        end
    end

    s = 4

    i::Int = 1
    t_exp = t
    @. U[i] = u
    lim!(U[i], p, t_exp, u)
    dss!(U[i], p, t_exp)
    T_lim!(T_lim[i], U[i], p, t_exp)
    T_exp!(T_exp[i], U[i], p, t_exp)

    i = 2
    t_exp = t + dt * c_exp[i]
    @. U[i] = u + dt * a_exp[i, 1] * T_lim[1]
    lim!(U[i], p, t_exp, u)
    @. U[i] += dt * a_exp[i, 1] * T_exp[1]
    dss!(U[i], p, t_exp)

    @. temp = U[i] # used in closures
    let i = i
        implicit_equation_residual! = (residual, Ui) -> begin
            T_imp!(residual, Ui, p, t + dt * c_imp[i]) #= t_imp =#
            @. residual = temp + dt * a_imp[i, i] * residual - Ui
        end
        implicit_equation_jacobian! =
            (jacobian, Ui) -> begin
                T_imp!.Wfact(jacobian, Ui, p, dt * a_imp[i, i], t + dt * c_imp[i]) #= t_imp =#
            end
        solve_newton!(
            newtons_method,
            newtons_method_cache,
            U[i],
            implicit_equation_residual!,
            implicit_equation_jacobian!,
        )
    end

    @. T_imp[i] = (U[i] - temp) / (dt * a_imp[i, i])

    T_lim!(T_lim[i], U[i], p, t_exp)
    T_exp!(T_exp[i], U[i], p, t_exp)

    i = 3
    t_exp = t + dt * c_exp[i]
    @. U[i] = u + dt * a_exp[i, 1] * T_lim[1] + dt * a_exp[i, 2] * T_lim[2]
    lim!(U[i], p, t_exp, u)
    @. U[i] += dt * a_exp[i, 1] * T_exp[1] + dt * a_exp[i, 2] * T_exp[2] + dt * a_imp[i, 2] * T_imp[2]
    dss!(U[i], p, t_exp)

    @. temp = U[i] # used in closures
    let i = i
        implicit_equation_residual! = (residual, Ui) -> begin
            T_imp!(residual, Ui, p, t + dt * c_imp[i]) #= t_imp =#
            @. residual = temp + dt * a_imp[i, i] * residual - Ui
        end
        implicit_equation_jacobian! =
            (jacobian, Ui) -> begin
                T_imp!.Wfact(jacobian, Ui, p, dt * a_imp[i, i], t + dt * c_imp[i]) #= t_imp =#
            end
        solve_newton!(
            newtons_method,
            newtons_method_cache,
            U[i],
            implicit_equation_residual!,
            implicit_equation_jacobian!,
        )
    end

    @. T_imp[i] = (U[i] - temp) / (dt * a_imp[i, i])

    T_lim!(T_lim[i], U[i], p, t_exp)
    T_exp!(T_exp[i], U[i], p, t_exp)
    i = 4
    t_exp = t + dt
    @. U[i] = u + dt * a_exp[i, 1] * T_lim[1] + dt * a_exp[i, 2] * T_lim[2] + dt * a_exp[i, 3] * T_lim[3]
    lim!(U[i], p, t_exp, u)
    @. U[i] +=
        dt * a_exp[i, 1] * T_exp[1] +
        dt * a_exp[i, 2] * T_exp[2] +
        dt * a_exp[i, 3] * T_exp[3] +
        dt * a_imp[i, 2] * T_imp[2] +
        dt * a_imp[i, 3] * T_imp[3]
    dss!(U[i], p, t_exp)

    @. temp = U[i] # used in closures
    let i = i
        implicit_equation_residual! = (residual, Ui) -> begin
            T_imp!(residual, Ui, p, t + dt * c_imp[i]) #= t_imp =#
            @. residual = temp + dt * a_imp[i, i] * residual - Ui
        end
        implicit_equation_jacobian! =
            (jacobian, Ui) -> begin
                T_imp!.Wfact(jacobian, Ui, p, dt * a_imp[i, i], t + dt * c_imp[i]) #= t_imp =#
            end
        solve_newton!(
            newtons_method,
            newtons_method_cache,
            U[i],
            implicit_equation_residual!,
            implicit_equation_jacobian!,
        )
    end

    @. T_imp[i] = (U[i] - temp) / (dt * a_imp[i, i])

    T_lim!(T_lim[i], U[i], p, t_exp)
    T_exp!(T_exp[i], U[i], p, t_exp)

    # final
    i = -1

    t_final = t + dt
    @. temp = u + dt * b_exp[2] * T_lim[2] + dt * b_exp[3] * T_lim[3] + dt * b_exp[4] * T_lim[4]
    lim!(temp, p, t_final, u)
    @. u =
        temp +
        dt * b_exp[2] * T_exp[2] +
        dt * b_exp[3] * T_exp[3] +
        dt * b_exp[4] * T_exp[4] +
        dt * b_imp[2] * T_imp[2] +
        dt * b_imp[3] * T_imp[3] +
        dt * b_imp[4] * T_imp[4]
    dss!(u, p, t_final)

    return u
end

function solve_newton_ars!(alg::NewtonsMethod, cache, x, f!, j! = nothing)
    (; max_iters, update_j, krylov_method, convergence_checker, verbose) = alg
    (; krylov_method_cache, convergence_checker_cache) = cache
    (; Δx, f, j) = cache
    if (!isnothing(j)) && needs_update!(update_j, NewNewtonSolve())
        j!(j, x)
    end
    for n in 0:max_iters
        # Update x[n] with Δx[n - 1], and exit the loop if Δx[n] is not needed.
        n > 0 && (x .-= Δx)
        if n == max_iters && isnothing(convergence_checker)
            is_verbose(verbose) && @info "Newton iteration $n: ‖x‖ = $(norm(x)), ‖Δx‖ = N/A"
            break
        end

        # Compute Δx[n].
        if (!isnothing(j)) && needs_update!(update_j, NewNewtonIteration())
            j!(j, x)
        end
        f!(f, x)
        if isnothing(krylov_method)
            ldiv!(Δx, j, f)
        else
            solve_krylov!(krylov_method, krylov_method_cache, Δx, x, f!, f, n, j)
        end
        is_verbose(verbose) && @info "Newton iteration $n: ‖x‖ = $(norm(x)), ‖Δx‖ = $(norm(Δx))"

        # Check for convergence if necessary.
        if !isnothing(convergence_checker)
            check_convergence!(convergence_checker, convergence_checker_cache, x, Δx, n) && break
            n == max_iters && @warn "Newton's method did not converge within $n iterations"
        end
    end
end
