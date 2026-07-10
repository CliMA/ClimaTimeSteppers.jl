#=
julia --project=test
using Revise; include("test/solvers/multirate_step_exchange.jl")
=#
using ClimaTimeSteppers, LinearAlgebra, Test
import ClimaTimeSteppers as CTS
import LinearAlgebra: mul!, norm

# Linear two-scale IMEX problem for the step-exchange multirate family.
#
# The 2x2 generators are built from the identity `Id2` and the rotation
# generator `J2`, which commute, so `exp((A_exp + A_imp + A_slow) t) Y0` is the
# exact solution. `A_exp` is the fast explicit part (sub-cycled), `A_imp` the
# fast implicit part (the IMEX inner Newton solve), and `A_slow` the slow
# relaxation (frozen forcing). The forcing freeze mirrors the split-explicit
# form: evaluate the full explicit tendency and subtract the sub-cycled part.
const J2 = [0.0 -1.0; 1.0 0.0]
const Id2 = [1.0 0.0; 0.0 1.0]
const A_exp = 2.0 .* J2
const A_imp = -1.0 .* Id2 .+ 1.0 .* J2
const A_slow = -0.3 .* Id2
const A_full_exp = A_exp .+ A_slow
const A_total = A_exp .+ A_imp .+ A_slow
const Y0 = [1.0, 0.5]

exact_solution(t) = exp(A_total .* t) * Y0

# Build a step-exchange `Multirate` integrator over the linear problem.
# `route_lim` routes the frozen slow forcing through the limited output `G_lim`
# (and the inner `lim!` path) instead of the unlimited output `G`; the counters
# record how often the slow forcing freeze and the limiter fire.
function step_exchange_integrator(outer; route_lim = false, dt, fast_dt, n_steps)
    freeze_count = Ref(0)
    lim_count = Ref(0)
    scratch = zeros(2)

    fast! = function (du_exp, du_lim, u, p, t)
        mul!(du_exp, A_exp, u)
        du_lim .= 0.0
        return nothing
    end
    T_imp! = CTS.ODEFunction(
        (du, u, p, t) -> mul!(du, A_imp, u);
        jac_prototype = zeros(2, 2),
        Wfact = (W, u, p, dtγ, t) -> (W .= dtγ .* A_imp .- Id2),
    )
    lim! = function (y, p, t, ref)
        lim_count[] += 1
        @. y = clamp(y, -1.0e3, 1.0e3)
        return nothing
    end
    f_fast = CTS.ClimaODEFunction(;
        T_exp_T_lim! = fast!,
        T_imp!,
        cache! = Returns(nothing),
        cache_imp! = Returns(nothing),
        lim!,
        dss! = Returns(nothing),
        initialize_imp! = Returns(nothing),
    )
    freeze! = function (G, G_lim, u, p, t)
        freeze_count[] += 1
        mul!(G, A_full_exp, u)
        mul!(scratch, A_exp, u)
        G .-= scratch
        if route_lim
            G_lim .= G
            G .= 0.0
        else
            G_lim .= 0.0
        end
        return nothing
    end

    prob = CTS.SplitODEProblem(f_fast, freeze!, copy(Y0), (0.0, dt * n_steps), nothing)
    inner = CTS.IMEXAlgorithm(CTS.ARS343(), CTS.NewtonsMethod(; max_iters = 2))
    integ = CTS.init(prob, CTS.Multirate(inner, outer); dt, fast_dt)
    return (; integ, freeze_count, lim_count)
end

function run_step_exchange(outer; kwargs...)
    (; integ, freeze_count, lim_count) =
        step_exchange_integrator(outer; kwargs...)
    n_steps = kwargs[:n_steps]
    for _ in 1:n_steps
        CTS.step!(integ)
    end
    return (; u = copy(integ.u), freeze_count = freeze_count[], lim_count = lim_count[])
end

# Least-squares slope of log(error) versus log(dt).
function convergence_slope(dts, errs)
    log_dts = log10.(dts)
    log_errs = log10.(errs)
    slope, _ = hcat(log_dts, ones(length(dts))) \ log_errs
    return slope
end

@testset "step-exchange multirate" begin
    @testset "convergence order" begin
        n_steps = [8, 16, 32, 64]
        t_end = 1.0
        dts = t_end ./ n_steps
        for (outer, expected) in
            ((LieSplitOuter(), 1), (TrapezoidalSplitOuter(), 2))
            errs = map(zip(dts, n_steps)) do (dt, ns)
                res = run_step_exchange(
                    outer;
                    dt,
                    fast_dt = dt / 8,
                    n_steps = ns,
                )
                norm(res.u .- exact_solution(t_end))
            end
            slope = convergence_slope(dts, errs)
            @test abs(slope - expected) < 0.2
        end
    end

    @testset "slow-function call count" begin
        n_steps = 5
        for (outer, per_step) in
            ((LieSplitOuter(), 1), (TrapezoidalSplitOuter(), 2))
            res = run_step_exchange(
                outer;
                dt = 0.1,
                fast_dt = 0.1 / 4,
                n_steps,
            )
            @test res.freeze_count == per_step * n_steps
        end
    end

    @testset "limited-forcing path" begin
        args = (; dt = 0.025, fast_dt = 0.025 / 4, n_steps = 20)
        base = run_step_exchange(LieSplitOuter(); route_lim = false, args...)
        limited = run_step_exchange(LieSplitOuter(); route_lim = true, args...)
        # The limiter fires in the inner sub-cycle for both routings.
        @test limited.lim_count > 0
        # Routing the frozen forcing through the limited output reproduces the
        # unlimited routing (the loose limiter never clips these states) and
        # converges to the exact solution.
        @test limited.u ≈ base.u rtol = 1.0e-10
        @test norm(limited.u .- exact_solution(0.5)) < 5.0e-3
    end
end
