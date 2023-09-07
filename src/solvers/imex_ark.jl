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

struct IMEXARKCache{SCU, SCE, SCI, T, Γ, NMC}
    U::SCU     # sparse container of length s
    T_lim::SCE # sparse container of length s
    T_exp::SCE # sparse container of length s
    T_imp::SCI # sparse container of length s
    temp::T
    γ::Γ
    newtons_method_cache::NMC
end

function init_cache(prob::DiffEqBase.AbstractODEProblem, alg::IMEXAlgorithm{Unconstrained}; kwargs...)
    (; u0, f) = prob
    (; T_imp!) = f
    (; tableau, newtons_method) = alg
    (; a_exp, b_exp, a_imp, b_imp) = tableau
    s = length(b_exp)
    inds = ntuple(i -> i, s)
    inds_T_exp = filter(i -> !all(iszero, a_exp[:, i]) || !iszero(b_exp[i]), inds)
    inds_T_imp = filter(i -> !all(iszero, a_imp[:, i]) || !iszero(b_imp[i]), inds)
    U = SparseContainer(map(i -> zero(u0), collect(1:length(inds))), inds)
    T_lim = SparseContainer(map(i -> zero(u0), collect(1:length(inds_T_exp))), inds_T_exp)
    T_exp = SparseContainer(map(i -> zero(u0), collect(1:length(inds_T_exp))), inds_T_exp)
    T_imp = SparseContainer(map(i -> zero(u0), collect(1:length(inds_T_imp))), inds_T_imp)
    temp = zero(u0)
    γs = unique(filter(!iszero, diag(a_imp)))
    γ = length(γs) == 1 ? γs[1] : nothing # TODO: This could just be a constant.
    jac_prototype = has_jac(T_imp!) ? T_imp!.jac_prototype : nothing
    newtons_method_cache =
        isnothing(T_imp!) || isnothing(newtons_method) ? nothing : allocate_cache(newtons_method, u0, jac_prototype)
    return IMEXARKCache(U, T_lim, T_exp, T_imp, temp, γ, newtons_method_cache)
end

step_u!(integrator, cache::IMEXARKCache) = step_u!(integrator, cache, integrator.alg.name)

include("hard_coded_ars343.jl")
# generic fallback
function step_u!(integrator, cache::IMEXARKCache, name)
    (; u, p, t, dt, sol, alg) = integrator
    (; f) = sol.prob
    (; T_lim!, T_exp!, T_imp!, lim!, dss!) = f
    (; name, tableau, newtons_method) = alg
    (; a_exp, b_exp, a_imp, b_imp, c_exp, c_imp) = tableau
    (; U, T_lim, T_exp, T_imp, temp, γ, newtons_method_cache) = cache
    s = length(b_exp)

    if !isnothing(T_imp!) && !isnothing(newtons_method)
        NVTX.@range "update!" color = colorant"yellow" begin
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
    end

    for i in 1:s
        NVTX.@range "stage" payload = i begin
            t_exp = t + dt * c_exp[i]
            t_imp = t + dt * c_imp[i]

            NVTX.@range "U[i] = u" color = colorant"yellow" begin
                @. U[i] = u
            end

            if !isnothing(T_lim!) # Update based on limited tendencies from previous stages
                NVTX.@range "U+=dt*a_exp*T_lim" color = colorant"yellow" begin
                    for j in 1:(i - 1)
                        iszero(a_exp[i, j]) && continue
                        @. U[i] += dt * a_exp[i, j] * T_lim[j]
                    end
                end
                NVTX.@range "lim!" color = colorant"yellow" begin
                    lim!(U[i], p, t_exp, u)
                end
            end

            if !isnothing(T_exp!) # Update based on explicit tendencies from previous stages
                NVTX.@range "U+=dt*a_exp*T_exp" color = colorant"yellow" begin
                    for j in 1:(i - 1)
                        iszero(a_exp[i, j]) && continue
                        @. U[i] += dt * a_exp[i, j] * T_exp[j]
                    end
                end
            end

            if !isnothing(T_imp!) # Update based on implicit tendencies from previous stages
                NVTX.@range "U+=dt*a_imp*T_imp" color = colorant"yellow" begin
                    for j in 1:(i - 1)
                        iszero(a_imp[i, j]) && continue
                        @. U[i] += dt * a_imp[i, j] * T_imp[j]
                    end
                end
            end

            NVTX.@range "dss!" color = colorant"yellow" begin
                dss!(U[i], p, t_exp)
            end

            if !isnothing(T_imp!) && !iszero(a_imp[i, i]) # Implicit solve
                @assert !isnothing(newtons_method)
                NVTX.@range "temp = U[i]" color = colorant"yellow" begin
                    @. temp = U[i]
                end
                # TODO: can/should we remove these closures?
                implicit_equation_residual! =
                    (residual, Ui) -> begin
                        NVTX.@range "T_imp!" color = colorant"yellow" begin
                            T_imp!(residual, Ui, p, t_imp)
                        end
                        NVTX.@range "residual=temp+dt*a_imp*residual-Ui" color = colorant"yellow" begin
                            @. residual = temp + dt * a_imp[i, i] * residual - Ui
                        end
                    end
                implicit_equation_jacobian! = (jacobian, Ui) -> T_imp!.Wfact(jacobian, Ui, p, dt * a_imp[i, i], t_imp)

                NVTX.@range "solve_newton!" color = colorant"yellow" begin
                    solve_newton!(
                        newtons_method,
                        newtons_method_cache,
                        U[i],
                        implicit_equation_residual!,
                        implicit_equation_jacobian!,
                    )
                end
            end

            # We do not need to DSS U[i] again because the implicit solve should
            # give the same results for redundant columns (as long as the implicit
            # tendency only acts in the vertical direction).

            if !all(iszero, a_imp[:, i]) || !iszero(b_imp[i])
                if !isnothing(T_imp!)
                    if iszero(a_imp[i, i])
                        # If its coefficient is 0, T_imp[i] is effectively being
                        # treated explicitly.
                        NVTX.@range "T_imp!" color = colorant"yellow" begin
                            T_imp!(T_imp[i], U[i], p, t_imp)
                        end
                    else
                        # If T_imp[i] is being treated implicitly, ensure that it
                        # exactly satisfies the implicit equation.
                        NVTX.@range "T_imp=(U-temp)/(dt*a_imp)" color = colorant"yellow" begin
                            @. T_imp[i] = (U[i] - temp) / (dt * a_imp[i, i])
                        end
                    end
                end
            end

            if !all(iszero, a_exp[:, i]) || !iszero(b_exp[i])
                if !isnothing(T_lim!)
                    NVTX.@range "T_lim!" color = colorant"yellow" begin
                        T_lim!(T_lim[i], U[i], p, t_exp)
                    end
                end
                if !isnothing(T_exp!)
                    NVTX.@range "T_exp!" color = colorant"yellow" begin
                        T_exp!(T_exp[i], U[i], p, t_exp)
                    end
                end
            end
        end
    end

    t_final = t + dt

    if !isnothing(T_lim!) # Update based on limited tendencies from previous stages
        NVTX.@range "temp=u" color = colorant"yellow" begin
            @. temp = u
        end
        NVTX.@range "temp+=dt*b_exp*T_lim" color = colorant"yellow" begin
            for j in 1:s
                iszero(b_exp[j]) && continue
                @. temp += dt * b_exp[j] * T_lim[j]
            end
        end
        NVTX.@range "lim!" color = colorant"yellow" begin
            lim!(temp, p, t_final, u)
        end
        NVTX.@range "u=temp" color = colorant"yellow" begin
            @. u = temp
        end
    end

    if !isnothing(T_exp!) # Update based on explicit tendencies from previous stages
        NVTX.@range "u+=dt*b_exp*T_exp" color = colorant"yellow" begin
            for j in 1:s
                iszero(b_exp[j]) && continue
                @. u += dt * b_exp[j] * T_exp[j]
            end
        end
    end

    if !isnothing(T_imp!) # Update based on implicit tendencies from previous stages
        NVTX.@range "u+=dt*b_imp*T_imp" color = colorant"yellow" begin
            for j in 1:s
                iszero(b_imp[j]) && continue
                @. u += dt * b_imp[j] * T_imp[j]
            end
        end
    end

    NVTX.@range "dss!" color = colorant"yellow" begin
        dss!(u, p, t_final)
    end

    return u
end
