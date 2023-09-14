struct IMEXSSPRKCache{U, SCI, B, Γ, NMC}
    U::U
    U_exp::U
    U_lim::U
    T_lim::U
    T_exp::U
    T_imp::SCI # sparse container of length s
    temp::U
    β::B
    γ::Γ
    newtons_method_cache::NMC
end

function init_cache(prob::DiffEqBase.AbstractODEProblem, alg::IMEXAlgorithm{SSP}; kwargs...)
    (; u0, f) = prob
    (; T_imp!) = f
    (; tableau, newtons_method) = alg
    (; a_exp, b_exp, a_imp, b_imp) = tableau
    s = length(b_exp)
    inds = ntuple(i -> i, s)
    inds_T_imp = filter(i -> !all(iszero, a_imp[:, i]) || !iszero(b_imp[i]), inds)
    U = similar(u0)
    U_exp = similar(u0)
    T_lim = similar(u0)
    T_exp = similar(u0)
    U_lim = similar(u0)
    T_imp = SparseContainer(map(i -> similar(u0), collect(1:length(inds_T_imp))), inds_T_imp)
    temp = similar(u0)
    â_exp = vcat(a_exp, b_exp')
    β = diag(â_exp, -1)
    for i in 1:length(β)
        if â_exp[(i + 1):end, i] != cumprod(β[i:end])
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
    jac_prototype = has_jac(T_imp!) ? T_imp!.jac_prototype : nothing
    newtons_method_cache =
        isnothing(T_imp!) || isnothing(newtons_method) ? nothing : allocate_cache(newtons_method, u0, jac_prototype)
    return IMEXSSPRKCache(U, U_exp, U_lim, T_lim, T_exp, T_imp, temp, β, γ, newtons_method_cache)
end

step_u!(integrator, cache::IMEXSSPRKCache) = step_u!(integrator, cache, integrator.alg.name)

function step_u!(integrator, cache::IMEXSSPRKCache, name)
    (; u, p, t, dt, sol, alg) = integrator
    (; f) = sol.prob
    (; T_lim!, T_exp!, T_imp!, lim!, dss!) = f
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

    for i in 1:s
        t_exp = t + dt * c_exp[i]
        t_imp = t + dt * c_imp[i]

        if i == 1
            @. U_exp = u
        elseif !iszero(β[i - 1])
            if !isnothing(T_lim!)
                @. U_lim = U_exp + dt * T_lim
                lim!(U_lim, p, t_exp, U_exp)
                @. U_exp = U_lim
            end
            if !isnothing(T_exp!)
                @. U_exp += dt * T_exp
            end
            @. U_exp = (1 - β[i - 1]) * u + β[i - 1] * U_exp
        end

        dss!(U_exp, p, t_exp)

        @. U = U_exp
        if !isnothing(T_imp!) # Update based on implicit tendencies from previous stages
            for j in 1:(i - 1)
                iszero(a_imp[i, j]) && continue
                @. U += dt * a_imp[i, j] * T_imp[j]
            end
        end

        if !isnothing(T_imp!) && !iszero(a_imp[i, i]) # Implicit solve
            @assert !isnothing(newtons_method)
            @. temp = U
            # TODO: can/should we remove these closures?
            implicit_equation_residual! = (residual, Ui) -> begin
                T_imp!(residual, Ui, p, t_imp)
                @. residual = temp + dt * a_imp[i, i] * residual - Ui
            end
            implicit_equation_jacobian! = (jacobian, Ui) -> T_imp!.Wfact(jacobian, Ui, p, dt * a_imp[i, i], t_imp)
            solve_newton!(
                newtons_method,
                newtons_method_cache,
                U,
                implicit_equation_residual!,
                implicit_equation_jacobian!,
            )
        end

        # We do not need to DSS U again because the implicit solve should
        # give the same results for redundant columns (as long as the implicit
        # tendency only acts in the vertical direction).

        if !all(iszero, a_imp[:, i]) || !iszero(b_imp[i])
            if !isnothing(T_imp!)
                if iszero(a_imp[i, i])
                    # If its coefficient is 0, T_imp[i] is effectively being
                    # treated explicitly.
                    T_imp!(T_imp[i], U, p, t_imp)
                else
                    # If T_imp[i] is being treated implicitly, ensure that it
                    # exactly satisfies the implicit equation.
                    @. T_imp[i] = (U - temp) / (dt * a_imp[i, i])
                end
            end
        end

        if !iszero(β[i])
            if !isnothing(T_lim!)
                T_lim!(T_lim, U, p, t_exp)
            end
            if !isnothing(T_exp!)
                T_exp!(T_exp, U, p, t_exp)
            end
        end
    end

    t_final = t + dt

    if !iszero(β[s])
        if !isnothing(T_lim!)
            @. U_lim = U_exp + dt * T_lim
            lim!(U_lim, p, t_final, U_exp)
            @. U_exp = U_lim
        end
        if !isnothing(T_exp!)
            @. U_exp += dt * T_exp
        end
        @. u = (1 - β[s]) * u + β[s] * U_exp
    end

    if !isnothing(T_imp!) # Update based on implicit tendencies from previous stages
        for j in 1:s
            iszero(b_imp[j]) && continue
            @. u += dt * b_imp[j] * T_imp[j]
        end
    end

    dss!(u, p, t_final)

    return u
end
