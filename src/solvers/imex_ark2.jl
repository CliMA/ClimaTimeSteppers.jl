export ClimaODEFunction, NewIMEXARKAlgorithm, NewARS343

Base.@kwdef struct ClimaODEFunction{TL, TE, TI, L, D, S} <: DiffEqBase.AbstractODEFunction{true}
    T_lim!::TL = nothing # nothing or (uₜ, u, p, t) -> ...
    T_exp!::TE = nothing # nothing or (uₜ, u, p, t) -> ...
    T_imp!::TI = nothing # nothing or (uₜ, u, p, t) -> ...
    lim!::L = (u, p, t, u_ref) -> nothing
    dss!::D = (u, p, t) -> nothing
    stage_callback!::S = (u, p, t) -> nothing
end

# Don't wrap a ClimaODEFunction in an ODEFunction (makes ODEProblem work).
DiffEqBase.ODEFunction{iip}(f::ClimaODEFunction) where {iip} = f
DiffEqBase.ODEFunction(f::ClimaODEFunction) = f

struct NewIMEXARKAlgorithm{VS <: StaticArrays.StaticArray, MS <: StaticArrays.StaticArray, NM} <: DistributedODEAlgorithm
    a_exp::MS # matrix of size s×s
    b_exp::VS # vector of length s
    c_exp::VS # vector of length s
    a_imp::MS # matrix of size s×s
    b_imp::VS # vector of length s
    c_imp::VS # vector of length s
    newtons_method::NM
end
NewIMEXARKAlgorithm(;
    a_exp,
    b_exp = a_exp[end, :],
    c_exp = vec(sum(a_exp; dims = 2)),
    a_imp,
    b_imp = a_imp[end, :],
    c_imp = vec(sum(a_imp; dims = 2)),
    newtons_method,
) = NewIMEXARKAlgorithm(a_exp, b_exp, c_exp, a_imp, b_imp, c_imp, newtons_method)

# TODO: make new package, ButcherTableaus.jl? Or just separate into separate file

function NewARS343(newtons_method)
    γ = 0.4358665215084590
    a42 = 0.5529291480359398
    a43 = a42
    b1 = -3/2 * γ^2 + 4 * γ - 1/4
    b2 =  3/2 * γ^2 - 5 * γ + 5/4
    a31 = (1 - 9/2 * γ + 3/2 * γ^2) * a42 +
        (11/4 - 21/2 * γ + 15/4 * γ^2) * a43 - 7/2 + 13 * γ - 9/2 * γ^2
    a32 = (-1 + 9/2 * γ - 3/2 * γ^2) * a42 +
        (-11/4 + 21/2 * γ - 15/4 * γ^2) * a43 + 4 - 25/2 * γ + 9/2 * γ^2
    a41 = 1 - a42 - a43
    return NewIMEXARKAlgorithm(;
        a_exp = @SArray([
            0   0   0   0;
            γ   0   0   0;
            a31 a32 0   0;
            a41 a42 a43 0;
        ]),
        b_exp = @SArray([0, b1, b2, γ]),
        a_imp = @SArray([
            0 0       0  0;
            0 γ       0  0;
            0 (1-γ)/2 γ  0;
            0 b1      b2 γ;
        ]),
        newtons_method,
    )
end

has_jac(T_imp!) =
    hasfield(typeof(T_imp!), :Wfact) &&
    hasfield(typeof(T_imp!), :jac_prototype) &&
    !isnothing(T_imp!.Wfact) &&
    !isnothing(T_imp!.jac_prototype)

struct NewIMEXARKCache{SCU, SCE, SCI, T, NMC}
    U::SCU     # sparse container of length s
    T_lim::SCE # sparse container of length s
    T_exp::SCE # sparse container of length s
    T_imp::SCI # sparse container of length s
    temp::T
    newtons_method_cache::NMC
end

function cache(prob::DiffEqBase.AbstractODEProblem, alg::NewIMEXARKAlgorithm; kwargs...)
    (; u0, f) = prob
    (; T_imp!) = f
    (; a_exp, b_exp, a_imp, b_imp, newtons_method) = alg
    s = length(b_exp)
    inds = ntuple(i->i, s)
    inds_T_exp = filter(i -> !all(iszero, a_exp[:, i]) || !iszero(b_exp[i]), inds)
    inds_T_imp = filter(i -> !all(iszero, a_imp[:, i]) || !iszero(b_imp[i]), inds)
    U = SparseContainer(map(i->similar(u0), collect(1:length(inds))), inds)
    T_lim = SparseContainer(map(i->similar(u0), collect(1:length(inds_T_exp))), inds_T_exp)
    T_exp = SparseContainer(map(i->similar(u0), collect(1:length(inds_T_exp))), inds_T_exp)
    T_imp = SparseContainer(map(i->similar(u0), collect(1:length(inds_T_imp))), inds_T_imp)
    temp = similar(u0)
    jac_prototype = has_jac(T_imp!) ? T_imp!.jac_prototype : nothing
    newtons_method_cache = allocate_cache(newtons_method, u0, jac_prototype)
    return NewIMEXARKCache(U, T_lim, T_exp, T_imp, temp, newtons_method_cache)
end

function step_u!(integrator, cache::NewIMEXARKCache)
    (; u, p, t, dt, sol, alg) = integrator
    (; f) = sol.prob
    (; T_lim!, T_exp!, T_imp!, lim!, dss!, stage_callback!) = f
    (; a_exp, b_exp, c_exp, a_imp, b_imp, c_imp, newtons_method) = alg
    (; U, T_lim, T_exp, T_imp, temp, newtons_method_cache) = cache
    s = length(b_exp)

    # TODO: Improve the update_j interface.
    if !isnothing(T_imp!) && !iszero(a_imp[end, end]) && has_jac(T_imp!)
        run!(
            newtons_method.update_j,
            newtons_method_cache.update_j_cache,
            NewStep(),
            (jacobian, u) ->
                T_imp!.Wfact(jacobian, u, p, dt * a_imp[end, end], t),
            newtons_method_cache.j,
            u,
        )
    end

    for i in 1:s
        t_exp = t + dt * c_exp[i]
        t_imp = t + dt * c_imp[i]

        @. U[i] = u

        if !isnothing(T_lim!) # Update based on limited tendencies from previous stages
            for j in 1:(i - 1)
                iszero(a_exp[i, j]) && continue
                @. U[i] += dt * a_exp[i, j] * T_lim[j]
            end
            lim!(U[i], p, t_exp, u)
        end

        if !isnothing(T_exp!) # Update based on explicit tendencies from previous stages
            for j in 1:(i - 1)
                iszero(a_exp[i, j]) && continue
                @. U[i] += dt * a_exp[i, j] * T_exp[j]
            end
        end

        if !isnothing(T_imp!) # Update based on implicit tendencies from previous stages
            for j in 1:(i - 1)
                iszero(a_imp[i, j]) && continue
                @. U[i] += dt * a_imp[i, j] * T_imp[j]
            end
        end

        dss!(U[i], p, t_exp)

        if !isnothing(T_imp!) && !iszero(a_imp[i, i]) # Implicit solve
            @. temp = U[i]
            # TODO: can/should we remove these closures?
            implicit_equation_residual! = (residual, Ui) -> begin
                T_imp!(residual, Ui, p, t_imp)
                @. residual = temp + dt * a_imp[i, i] * residual - Ui
            end
            implicit_equation_jacobian! =
                if has_jac(T_imp!)
                    (jacobian, Ui) ->
                        T_imp!.Wfact(jacobian, Ui, p, dt * a_imp[i, i], t_imp)
                else
                    nothing
                end
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
