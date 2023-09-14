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
            NVTX.@range "Wfact" color = colorant"orange" begin
                T_imp!.Wfact(jacobian, u, p, dt * γ, t)
            end
        end
    end

    s = 4

    NVTX.@range "Stage 1" color = colorant"brown" begin
        i::Int = 1
        t_exp = t
        NVTX.@range "U[i] = u" begin
            @. U[i] = u
        end
        NVTX.@range "lim!" color = colorant"blue" begin
            lim!(U[i], p, t_exp, u)
        end
        NVTX.@range "dss!" color = colorant"green" begin
            dss!(U[i], p, t_exp)
        end
        NVTX.@range "T_lim!" color = colorant"red" begin
            T_lim!(T_lim[i], U[i], p, t_exp)
        end
        NVTX.@range "T_exp!" color = colorant"purple" begin
            T_exp!(T_exp[i], U[i], p, t_exp)
        end

        i = 2
        t_exp = t + dt * c_exp[i]
        NVTX.@range "U=u+dt*a_exp*T_lim" begin
            @. U[i] = u + dt * a_exp[i, 1] * T_lim[1]
        end
        NVTX.@range "lim!" color = colorant"blue" begin
            lim!(U[i], p, t_exp, u)
        end
        NVTX.@range "U+=dt*a_exp*T_exp" begin
            @. U[i] += dt * a_exp[i, 1] * T_exp[1]
        end
        NVTX.@range "dss!" color = colorant"green" begin
            dss!(U[i], p, t_exp)
        end
    end
    NVTX.@range "Stage 2" color = colorant"brown" begin
        NVTX.@range "temp = U[i]" begin
            @. temp = U[i] # used in closures
        end
        let i = i
            implicit_equation_residual! =
                (residual, Ui) -> begin
                    NVTX.@range "T_imp!" color = colorant"pink" begin
                        T_imp!(residual, Ui, p, t + dt * c_imp[i]) #= t_imp =#
                    end
                    NVTX.@range "residual=temp+dt*a_imp*residual-Ui" begin
                        @. residual = temp + dt * a_imp[i, i] * residual - Ui
                    end
                end
            implicit_equation_jacobian! =
                (jacobian, Ui) -> begin
                    NVTX.@range "Wfact" color = colorant"orange" begin
                        T_imp!.Wfact(jacobian, Ui, p, dt * a_imp[i, i], t + dt * c_imp[i]) #= t_imp =#
                    end
                end
            solve_newton!(
                newtons_method,
                newtons_method_cache,
                U[i],
                implicit_equation_residual!,
                implicit_equation_jacobian!,
            )
        end

        NVTX.@range "T_imp=(U-temp)/(dt*a_imp)" begin
            @. T_imp[i] = (U[i] - temp) / (dt * a_imp[i, i])
        end

        NVTX.@range "T_lim!" color = colorant"red" begin
            T_lim!(T_lim[i], U[i], p, t_exp)
        end
        NVTX.@range "T_exp!" color = colorant"purple" begin
            T_exp!(T_exp[i], U[i], p, t_exp)
        end

        i = 3
        t_exp = t + dt * c_exp[i]
        NVTX.@range "U=u+dt*..." begin
            @. U[i] = u + dt * a_exp[i, 1] * T_lim[1] + dt * a_exp[i, 2] * T_lim[2]
        end
        NVTX.@range "lim!" color = colorant"blue" begin
            lim!(U[i], p, t_exp, u)
        end
        NVTX.@range "U=u+dt*..." begin
            @. U[i] += dt * a_exp[i, 1] * T_exp[1] + dt * a_exp[i, 2] * T_exp[2] + dt * a_imp[i, 2] * T_imp[2]
        end
        NVTX.@range "dss!" color = colorant"green" begin
            dss!(U[i], p, t_exp)
        end
    end

    NVTX.@range "Stage 3" color = colorant"brown" begin
        NVTX.@range "temp = U[i]" begin
            @. temp = U[i] # used in closures
        end
        let i = i
            implicit_equation_residual! =
                (residual, Ui) -> begin
                    NVTX.@range "T_imp!" color = colorant"pink" begin
                        T_imp!(residual, Ui, p, t + dt * c_imp[i]) #= t_imp =#
                    end
                    NVTX.@range "residual=temp+dt*a_imp*residual-Ui" begin
                        @. residual = temp + dt * a_imp[i, i] * residual - Ui
                    end
                end
            implicit_equation_jacobian! =
                (jacobian, Ui) -> begin
                    NVTX.@range "Wfact" color = colorant"orange" begin
                        T_imp!.Wfact(jacobian, Ui, p, dt * a_imp[i, i], t + dt * c_imp[i]) #= t_imp =#
                    end
                end
            solve_newton!(
                newtons_method,
                newtons_method_cache,
                U[i],
                implicit_equation_residual!,
                implicit_equation_jacobian!,
            )
        end

        NVTX.@range "T_imp=(U-temp)/(dt*a_imp)" begin
            @. T_imp[i] = (U[i] - temp) / (dt * a_imp[i, i])
        end

        NVTX.@range "T_lim!" color = colorant"red" begin
            T_lim!(T_lim[i], U[i], p, t_exp)
        end
        NVTX.@range "T_exp!" color = colorant"purple" begin
            T_exp!(T_exp[i], U[i], p, t_exp)
        end
        i = 4
        t_exp = t + dt
        NVTX.@range "U=u+dt*..." begin
            @. U[i] = u + dt * a_exp[i, 1] * T_lim[1] + dt * a_exp[i, 2] * T_lim[2] + dt * a_exp[i, 3] * T_lim[3]
        end
        NVTX.@range "lim!" color = colorant"blue" begin
            lim!(U[i], p, t_exp, u)
        end
        NVTX.@range "U+=dt*..." begin
            @. U[i] +=
                dt * a_exp[i, 1] * T_exp[1] +
                dt * a_exp[i, 2] * T_exp[2] +
                dt * a_exp[i, 3] * T_exp[3] +
                dt * a_imp[i, 2] * T_imp[2] +
                dt * a_imp[i, 3] * T_imp[3]
        end
        NVTX.@range "dss!" color = colorant"green" begin
            dss!(U[i], p, t_exp)
        end
    end
    NVTX.@range "Stage 4" color = colorant"brown" begin
        NVTX.@range "temp = U[i]" begin
            @. temp = U[i] # used in closures
        end
        let i = i
            implicit_equation_residual! =
                (residual, Ui) -> begin
                    NVTX.@range "T_imp!" color = colorant"pink" begin
                        T_imp!(residual, Ui, p, t + dt * c_imp[i]) #= t_imp =#
                    end
                    NVTX.@range "residual=temp+dt*a_imp*residual-Ui" begin
                        @. residual = temp + dt * a_imp[i, i] * residual - Ui
                    end
                end
            implicit_equation_jacobian! =
                (jacobian, Ui) -> begin
                    NVTX.@range "Wfact" color = colorant"orange" begin
                        T_imp!.Wfact(jacobian, Ui, p, dt * a_imp[i, i], t + dt * c_imp[i]) #= t_imp =#
                    end
                end
            solve_newton!(
                newtons_method,
                newtons_method_cache,
                U[i],
                implicit_equation_residual!,
                implicit_equation_jacobian!,
            )
        end

        NVTX.@range "T_imp=(U-temp)/(dt*a_imp)" begin
            @. T_imp[i] = (U[i] - temp) / (dt * a_imp[i, i])
        end

        NVTX.@range "T_lim!" color = colorant"red" begin
            T_lim!(T_lim[i], U[i], p, t_exp)
        end
        NVTX.@range "T_exp!" color = colorant"purple" begin
            T_exp!(T_exp[i], U[i], p, t_exp)
        end

        # final
        i = -1

        t_final = t + dt
        NVTX.@range "temp=u+dt*..." begin
            @. temp = u + dt * b_exp[2] * T_lim[2] + dt * b_exp[3] * T_lim[3] + dt * b_exp[4] * T_lim[4]
        end
        NVTX.@range "lim!" color = colorant"blue" begin
            lim!(temp, p, t_final, u)
        end
        NVTX.@range "u=temp+..." begin
            @. u =
                temp +
                dt * b_exp[2] * T_exp[2] +
                dt * b_exp[3] * T_exp[3] +
                dt * b_exp[4] * T_exp[4] +
                dt * b_imp[2] * T_imp[2] +
                dt * b_imp[3] * T_imp[3] +
                dt * b_imp[4] * T_imp[4]
        end
        NVTX.@range "dss!" color = colorant"green" begin
            dss!(u, p, t_final)
        end
    end
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
