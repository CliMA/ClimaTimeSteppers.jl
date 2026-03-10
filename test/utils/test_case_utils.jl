# ============================================================================ #
# IntegratorTestCase infrastructure (used by IMEX ARK, SSP, Rosenbrock tests
# and integration/integrator.jl)
# ============================================================================ #

reverse_problem(prob, analytic_sol) =
    ODEProblem(prob.f, analytic_sol(prob.tspan[2]), reverse(prob.tspan), prob.p)

struct IntegratorTestCase{FT, A, P, SP}
    test_name::String
    linear_implicit::Bool
    t_end::FT
    analytic_sol::A
    prob::P
    split_prob::SP
    default_num_steps::Int
    high_order_sample_shifts::Int
end

function ClimaIntegratorTestCase(;
    test_name,
    linear_implicit,
    t_end,
    Y₀,
    analytic_sol,
    tendency!,
    implicit_tendency! = nothing,
    explicit_tendency! = nothing,
    Wfact!,
    tgrad! = nothing,
    default_num_steps = 100,
    high_order_sample_shifts = 1,
)
    FT = typeof(t_end)
    jac_prototype = Matrix{FT}(undef, length(Y₀), length(Y₀))
    func_args = (; jac_prototype, Wfact = Wfact!, tgrad = tgrad!)
    tendency_func = ClimaODEFunction(; T_imp! = ODEFunction(tendency!; func_args...))

    T_imp! = if isnothing(implicit_tendency!)
        ODEFunction(tendency!; func_args...)
    else
        ODEFunction(implicit_tendency!; func_args...)
    end
    split_tendency_func = ClimaODEFunction(; T_exp! = explicit_tendency!, T_imp!)
    make_prob(func) = ODEProblem(func, Y₀, (FT(0), t_end), nothing)
    IntegratorTestCase(
        test_name,
        linear_implicit,
        t_end,
        analytic_sol,
        make_prob(tendency_func),
        make_prob(split_tendency_func),
        default_num_steps,
        high_order_sample_shifts,
    )
end

function clima_constant_tendency_test(::Type{FT}) where {FT}
    tendency = FT[1, 2, 3]
    ClimaIntegratorTestCase(;
        test_name = "constant_tendency",
        linear_implicit = true,
        t_end = FT(1),
        Y₀ = FT[0, 0, 0],
        analytic_sol = (t) -> tendency .* t,
        tendency! = (Yₜ, Y, _, t) -> Yₜ .= tendency,
        Wfact! = (W, Y, _, Δt, t) -> W .= -1,
    )
end
