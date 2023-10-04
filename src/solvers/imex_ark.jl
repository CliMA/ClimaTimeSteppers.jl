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
    U = zero(u0)
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

# generic fallback
function step_u!(integrator, cache::IMEXARKCache)
    (; u, p, t, dt, alg) = integrator
    (; f) = integrator.sol.prob
    (; post_explicit!, post_implicit!) = f
    (; T_lim!, T_exp!, T_imp!, lim!, dss!) = f
    (; tableau, newtons_method) = alg
    (; a_exp, b_exp, a_imp, b_imp, c_exp, c_imp) = tableau
    (; U, T_lim, T_exp, T_imp, temp, γ, newtons_method_cache) = cache
    s = length(b_exp)

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

    for i in 1:s
        t_exp = t + dt * c_exp[i]
        t_imp = t + dt * c_imp[i]

        @. U = u

        if !isnothing(T_lim!) # Update based on limited tendencies from previous stages
            for j in 1:(i - 1)
                iszero(a_exp[i, j]) && continue
                @. U += dt * a_exp[i, j] * T_lim[j]
            end
            lim!(U, p, t_exp, u)
        end

        if !isnothing(T_exp!) # Update based on explicit tendencies from previous stages
            for j in 1:(i - 1)
                iszero(a_exp[i, j]) && continue
                @. U += dt * a_exp[i, j] * T_exp[j]
            end
        end

        if !isnothing(T_imp!) # Update based on implicit tendencies from previous stages
            for j in 1:(i - 1)
                iszero(a_imp[i, j]) && continue
                @. U += dt * a_imp[i, j] * T_imp[j]
            end
        end

        dss!(U, p, t_exp)

        if !(!isnothing(T_imp!) && !iszero(a_imp[i, i])) # Implicit solve
            post_explicit!(U, p, t_imp)
        else
            @assert !isnothing(newtons_method)
            @. temp = U
            post_explicit!(U, p, t_imp)
            # TODO: can/should we remove these closures?
            implicit_equation_residual! = (residual, Ui) -> begin
                T_imp!(residual, Ui, p, t_imp)
                @. residual = temp + dt * a_imp[i, i] * residual - Ui
            end
            implicit_equation_jacobian! = (jacobian, Ui) -> T_imp!.Wfact(jacobian, Ui, p, dt * a_imp[i, i], t_imp)
            call_post_implicit! = Ui -> begin
                post_implicit!(Ui, p, t_imp)
            end
            call_post_implicit_last! =
                Ui -> begin
                    if (!all(iszero, a_imp[:, i]) || !iszero(b_imp[i])) && !iszero(a_imp[i, i])
                        # If T_imp[i] is being treated implicitly, ensure that it
                        # exactly satisfies the implicit equation.
                        @. T_imp[i] = (Ui - temp) / (dt * a_imp[i, i])
                    end
                    post_implicit!(Ui, p, t_imp)
                end

            solve_newton!(
                newtons_method,
                newtons_method_cache,
                U,
                implicit_equation_residual!,
                implicit_equation_jacobian!,
                call_post_implicit!,
                call_post_implicit_last!,
            )
        end

        # We do not need to DSS U again because the implicit solve should
        # give the same results for redundant columns (as long as the implicit
        # tendency only acts in the vertical direction).

        if !all(iszero, a_imp[:, i]) || !iszero(b_imp[i])
            if iszero(a_imp[i, i]) && !isnothing(T_imp!)
                # If its coefficient is 0, T_imp[i] is effectively being
                # treated explicitly.
                T_imp!(T_imp[i], U, p, t_imp)
            end
        end

        if !all(iszero, a_exp[:, i]) || !iszero(b_exp[i])
            if !isnothing(T_lim!)
                T_lim!(T_lim[i], U, p, t_exp)
            end
            if !isnothing(T_exp!)
                T_exp!(T_exp[i], U, p, t_exp)
            end
        end
    end

    t_final = t + dt

    if !isnothing(T_lim!) # Update based on limited tendencies from previous stages
        @. temp = u
        for j in 1:s
            iszero(b_exp[j]) && continue
            @. temp += dt * b_exp[j] * T_lim[j]
        end
        lim!(temp, p, t_final, u)
        @. u = temp
    end

    if !isnothing(T_exp!) # Update based on explicit tendencies from previous stages
        for j in 1:s
            iszero(b_exp[j]) && continue
            @. u += dt * b_exp[j] * T_exp[j]
        end
    end

    if !isnothing(T_imp!) # Update based on implicit tendencies from previous stages
        for j in 1:s
            iszero(b_imp[j]) && continue
            @. u += dt * b_imp[j] * T_imp[j]
        end
    end

    dss!(u, p, t_final)
    post_explicit!(u, p, t_final)

    return u
end
