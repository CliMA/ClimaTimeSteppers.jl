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

struct IMEXARKCache{SCE, SCI, T, Γ, NMC}
    U::T
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

step_u!(integrator, cache::IMEXARKCache) = step_u!(integrator, cache, integrator.alg.name)

include("hard_coded_ars343.jl")
# generic fallback
function step_u!(integrator, cache::IMEXARKCache, name)
    (; u, p, t, dt, sol, alg) = integrator
    (; f) = sol.prob
    (; T_lim!, T_exp!, T_imp!, lim!, dss!) = f
    (; tableau, newtons_method) = alg
    (; a_exp, b_exp, a_imp, b_imp, c_exp, c_imp) = tableau
    (; U, T_lim, T_exp, T_imp, temp, γ, newtons_method_cache) = cache
    Nstages = length(b_exp) - 1

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

    U .= u

    for stage in 1:Nstages
        NVTX.@range "explicit update" begin
            t_exp = t + dt * c_exp[stage]
            if !isnothing(T_lim!)
                NVTX.@range "compute limited tendency" color = colorant"yellow" begin
                    T_lim!(T_lim[stage], U, p, t_exp)
                end
            end
            if !isnothing(T_exp!)
                NVTX.@range "compute remaining tendency" color = colorant"blue" begin
                    T_exp!(T_exp[stage], U, p, t_exp)
                end
            end
            NVTX.@range "update U" color = colorant"green" begin
                U .= u
                for j in 1:stage
                    iszero(a_exp[stage + 1, j]) && continue
                    @. U += dt * a_exp[stage + 1, j] * T_lim[j]
                end
                if !isnothing(lim!)
                    NVTX.@range "apply limiter" color = colorant"yellow" begin
                        lim!(U, p, t_final, u)
                    end
                end
                for j in 1:stage
                    iszero(a_exp[stage + 1, j]) && continue
                    @. U += dt * a_exp[stage, j] * T_exp[j]
                end
                # TODO: convert to generic explicit callback
                NVTX.@range "dss!" color = colorant"yellow" begin
                    dss!(U, p, t_exp)
                end
            end
        end

        NVTX.@range "implicit update" begin
            t_imp = t + dt * c_imp[stage + 1]
            if iszero(a_imp[stage + 1, stage + 1]) # Implicit solve
                @assert !isnothing(newtons_method)
                @. temp = U
                # TODO: can/should we remove these closures?
                function implicit_equation_residual!(residual, U)
                    NVTX.@range "T_imp!" color = colorant"yellow" begin
                        T_imp!(residual, U, p, t_imp)
                    end
                    NVTX.@range "residual=temp+dt*a_imp*residual-Ui" color = colorant"yellow" begin
                        @. residual = temp + dt * a_imp[stage + 1, stage + 1] * residual - U
                    end
                end
                function implicit_equation_jacobian!(jacobian, U)
                    T_imp!.Wfact(jacobian, U, p, dt * a_imp[stage + 1, stage + 1], t_imp)
                end

                NVTX.@range "solve_newton!" color = colorant"yellow" begin
                    # TODO: add option for callback
                    solve_newton!(
                        newtons_method,
                        newtons_method_cache,
                        U,
                        implicit_equation_residual!,
                        implicit_equation_jacobian!,
                    )
                end
                @. T_imp[stage] = (U - temp) / (dt * a_imp[stage + 1, stage + 1])
            else
                T_imp!(T_imp[stage], U, p, t_imp)
            end
        end
    end

    NVTX.@range "final explicit update" begin
        t_final = t + dt
        NVTX.@range "compute limited tendency" color = colorant"yellow" begin
            T_lim!(T_lim[Nstages + 1], U, p, t_final)
        end
        NVTX.@range "compute remaining tendency" color = colorant"blue" begin
            T_exp!(T_exp[Nstages + 1], U, p, t_final)
        end
        NVTX.@range "update U" color = colorant"green" begin
            U .= u
            for j in 1:stage
                iszero(a_exp[stage + 1, j]) && continue
                @. U += dt * a_exp[stage + 1, j] * T_lim[j]
            end
            NVTX.@range "apply limiter" color = colorant"yellow" begin
                lim!(U, p, t_final, u)
            end
            for j in 1:stage
                iszero(a_exp[stage + 1, j]) && continue
                @. U += dt * a_exp[stage, j] * T_exp[j]
            end
            # TODO: convert to generic explicit callback
            NVTX.@range "dss!" color = colorant"yellow" begin
                dss!(U, p, t_final)
            end
        end
        u .= U
    end
    return u
end
