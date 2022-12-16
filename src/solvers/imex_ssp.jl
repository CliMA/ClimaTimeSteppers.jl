#=
U[i] = (1 - β[i-1]) * u + β[i-1] * (U[i-1] + dt * T_exp(U[i-1])) for i > 1 ==>
    U[2] = (1 - β[1]) * u + β[1] * (U[1] + dt * T_exp(U[1])) =
           u + dt * β[1] * T_exp(U[1])
    U[3] = (1 - β[2]) * u + β[2] * (U[2] + dt * T_exp(U[2])) =
           u +
           dt * β[1] * β[2] * T_exp(U[1]) +
           dt * β[2] * T_exp(U[2])
    U[4] = (1 - β[3]) * u + β[3] * (U[3] + dt * T_exp(U[3])) =
           u +
           dt * β[1] * β[2] * Β[3] * T_exp(U[1]) +
           dt * β[2] * Β[3] * T_exp(U[2]) +
           dt * Β[3] * T_exp(U[3])
    ...

U[i] = u + ∑_{j = 1}^{i - 1} dt * a_exp[i, j] * T_exp(U[j]) ==>
    a_exp = [
        0                  0           0    …
        β[1]               0           0    …
        β[1] * β[2]        β[2]        0    …
        β[1] * β[2] * β[3] β[2] * β[3] β[3] …
        ⋮                  ⋮            ⋮    ⋱
    ] ==>
    a_exp[i+1:s+1, i] = cumprod(β[i:s])
=#

"""
    IMEXSSPRKAlgorithm(
        tabname::AbstractTableau,
        newtons_method
    ) <: DistributedODEAlgorithm

A generic implementation of an IMEX SSP RK algorithm that can handle arbitrary
Butcher tableaus.
"""
struct IMEXSSPRKAlgorithm{B, T <: IMEXARKTableau, NM} <: DistributedODEAlgorithm
    β::B
    tab::T
    newtons_method::NM
end
function IMEXSSPRKAlgorithm(tabname::AbstractTableau, newtons_method)
    tab = tableau(tabname)
    (; a_exp, b_exp) = tab
    â_exp = vcat(a_exp, b_exp')
    β = diag(â_exp, -1)
    for i in 1:length(β)
        if â_exp[(i+1):end, i] != cumprod(β[i:end])
            error("Tableau does not satisfy requirements for an SSP RK method")
        end
    end
    IMEXSSPRKAlgorithm(β, tab, newtons_method)
end

struct IMEXSSPRKCache{U, SCE, SCI, T, Γ, NMC}
    U::U
    U_exp::U
    U_lim::U
    T_lim::U
    T_exp::U
    T_imp::SCI # sparse container of length s
    temp::T
    γ::Γ
    newtons_method_cache::NMC
end

function cache(prob::DiffEqBase.AbstractODEProblem, alg::IMEXSSPRKAlgorithm; kwargs...)
    (; u0, f) = prob
    (; T_imp!) = f
    (; tab, newtons_method) = alg
    (; a_exp, b_exp, a_imp, b_imp) = tab
    s = length(b_exp)
    inds = ntuple(i->i, s)
    inds_T_imp = filter(i -> !all(iszero, a_imp[:, i]) || !iszero(b_imp[i]), inds)
    U = similar(u0)
    U_exp = similar(u0)
    T_lim = similar(u0)
    T_exp = similar(u0)
    U_lim = similar(u0)
    T_imp = SparseContainer(map(i->similar(u0), collect(1:length(inds_T_imp))), inds_T_imp)
    temp = similar(u0)
    γs = unique(filter(!iszero, diag(a_imp)))
    γ = length(γs) == 1 ? γs[1] : nothing # TODO: This could just be a constant.
    jac_prototype = has_jac(T_imp!) ? T_imp!.jac_prototype : nothing
    newtons_method_cache = allocate_cache(newtons_method, u0, jac_prototype)
    return IMEXSSPRKCache(U, U_exp, U_lim, T_lim, T_exp, T_imp, temp, γ, newtons_method_cache)
end

function step_u!(integrator, cache::IMEXSSPRKCache)
    (; u, p, t, dt, sol, alg) = integrator
    (; f) = sol.prob
    (; T_lim!, T_exp!, T_imp!, lim!, dss!, stage_callback!) = f
    (; tab, newtons_method) = alg
    (; a_exp, b_exp, a_imp, b_imp, c_exp, c_imp) = tab
    (; U, U_lim, U_exp, T_lim, T_exp, T_imp, temp, γ, newtons_method_cache) = cache
    s = length(b_exp)

    if !isnothing(T_imp!)
        update!(
            newtons_method,
            newtons_method_cache,
            NewTimeStep(t),
            jacobian -> isnothing(γ) ?
                error(
                    "The tableau does not specify a unique value of γ for the \
                     duration of each time step; do not update based on the \
                     NewTimeStep signal when using this tableau."
                ) : T_imp!.Wfact(jacobian, u, p, dt * γ, t),
        )
    end

    for i in 1:s
        if i == 1
            U_exp .= u
        else
            if !isnothing(T_lim!)
                T_lim!(T_lim, U, p, t + dt * c_exp[i - 1])
                @. U_lim = U_exp + dt * β[i - 1] * T_lim
                lim!(U_lim, p, U_exp)
            else
                @. U_lim = U_exp # TODO: unnecessary
            end
            if !isnothing(T_exp!)
                T_exp!(T_exp, U, p, t + dt * c_exp[i - 1])
                @. U_exp = (1 - β[i - 1]) * u + β[i - 1] * (U_lim + dt * T_exp[i - 1])
            else
                @. U_exp = U_lim # TODO: unnecessary
            end
        end

        dss!(U_exp, p)

        
        if !isnothing(T_imp!) # Update based on implicit tendencies from previous stages
            for j in 1:(i - 1)
                iszero(a_imp[i, j]) && continue
                @. U[i] += dt * a_imp[i, j] * T_imp[j]
            end
        end

        if !isnothing(T_imp!) && !iszero(a_imp[i, i]) # Implicit solve
            @. temp = U[i]
            # TODO: can/should we remove these closures?
            implicit_equation_residual! = (residual, Ui) -> begin
                T_imp!(residual, Ui, p, t_imp)
                @. residual = temp + dt * a_imp[i, i] * residual - Ui
            end
            implicit_equation_jacobian! =
                (jacobian, Ui) ->
                    T_imp!.Wfact(jacobian, Ui, p, dt * a_imp[i, i], t_imp)
            run!(
                newtons_method,
                newtons_method_cache,
                U[i],
                implicit_equation_residual!,
                implicit_equation_jacobian!,
            )
        end

        # We do not need to DSS U[i] again because the implicit solve should
        # give the same results for redundant columns (as long as the implicit
        # tendency only acts in the vertical direction).

        if !all(iszero, a_imp[:, i]) || !iszero(b_imp[i])
            if !isnothing(T_imp!)
                if iszero(a_imp[i, i])
                    # If its coefficient is 0, T_imp[i] is effectively being
                    # treated explicitly.
                    T_imp!(T_imp[i], U[i], p, t_imp)
                else
                    # If T_imp[i] is being treated implicitly, ensure that it
                    # exactly satisfies the implicit equation.
                    @. T_imp[i] = (U[i] - temp) / (dt * a_imp[i, i])
                end
            end
        end

        stage_callback!(U[i], p, t_exp)

        if !all(iszero, a_exp[:, i]) || !iszero(b_exp[i])
            if !isnothing(T_lim!)
                T_lim!(T_lim[i], U[i], p, t_exp)
            end
            if !isnothing(T_exp!)
                T_exp!(T_exp[i], U[i], p, t_exp)
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

    return u
end
