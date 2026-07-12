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
# record how often the slow-forcing freeze and the limiter are called. When
# `log` is a vector, each `freeze!`, `cache!`, and `cache_imp!` call appends
# `(; hook, t, u)` with a copy of the state it was given.
function step_exchange_integrator(
    outer;
    route_lim = false,
    constrain_state! = Returns(nothing),
    log = nothing,
    dt,
    fast_dt,
    n_steps,
)
    freeze_count = Ref(0)
    lim_count = Ref(0)
    scratch = zeros(2)
    record(hook) = function (u, p, t)
        isnothing(log) || push!(log, (; hook, t, u = copy(u)))
        return nothing
    end

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
        cache! = record(:cache),
        cache_imp! = record(:cache_imp),
        lim!,
        dss! = Returns(nothing),
        constrain_state!,
        initialize_imp! = Returns(nothing),
    )
    freeze! = function (G, G_lim, u, p, t)
        freeze_count[] += 1
        isnothing(log) || push!(log, (; hook = :freeze, t, u = copy(u)))
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
        @test limited.lim_count > 0
        # Limited routing reproduces the unlimited routing.
        @test limited.u ≈ base.u rtol = 1.0e-10
        @test norm(limited.u .- exact_solution(args.dt * args.n_steps)) < 5.0e-3
    end

    @testset "complement sequencing" begin
        dt = 0.1
        fast_dt = dt / 4
        n_steps = 3
        shift = [0.05, -0.02]

        lie_calls = Tuple{Float64, Float64}[]
        lie_complement = function (u, p, t, dt)
            push!(lie_calls, (t, dt))
            @. u += shift * dt
            return nothing
        end
        lie = run_step_exchange(LieSplitOuter(lie_complement); dt, fast_dt, n_steps)
        lie_ref = run_step_exchange(LieSplitOuter(); dt, fast_dt, n_steps)
        # Lie applies the complement once per step over the full step.
        @test length(lie_calls) == n_steps
        @test all(c -> c[2] == dt, lie_calls)
        @test lie_calls[1][1] == 0.0
        @test lie.u != lie_ref.u

        trap_calls = Tuple{Float64, Float64}[]
        trap_complement = function (u, p, t, dt)
            push!(trap_calls, (t, dt))
            @. u += shift * dt
            return nothing
        end
        trap =
            run_step_exchange(TrapezoidalSplitOuter(trap_complement); dt, fast_dt, n_steps)
        trap_ref = run_step_exchange(TrapezoidalSplitOuter(); dt, fast_dt, n_steps)
        # Trapezoidal brackets each step with two half-step complement calls.
        @test length(trap_calls) == 2 * n_steps
        @test all(c -> c[2] == dt / 2, trap_calls)
        @test trap_calls[1][1] == 0.0
        @test trap_calls[2][1] == dt / 2
        @test trap.u != trap_ref.u
    end

    @testset "trapezoidal second-pass cache refresh" begin
        dt = 0.1
        log = []
        (; integ) = step_exchange_integrator(
            TrapezoidalSplitOuter();
            log,
            dt,
            fast_dt = dt / 4,
            n_steps = 1,
        )
        CTS.step!(integ)
        freezes = findall(e -> e.hook == :freeze, log)
        @test length(freezes) == 2
        @test log[freezes[1]].t == 0.0
        @test log[freezes[1]].u == Y0
        @test log[freezes[2]].t == dt
        full = findall(e -> e.hook == :cache, log)
        @test length(full) == 2
        mid, final = full
        # A full cache refresh occurs between the second-pass freeze and the
        # second sub-cycle, paired with the second-pass restart state and time.
        @test freezes[2] < mid
        @test log[mid].t == 0.0
        @test log[mid].u == Y0
        restart = findnext(e -> e.hook == :cache_imp, log, mid + 1)
        @test restart !== nothing && restart < final
        @test log[restart].t == 0.0
        @test log[restart].u == log[mid].u
        @test log[final].t == dt
        @test log[final].u == integ.u
    end

    @testset "trapezoidal second-complement cache refresh" begin
        dt = 0.1
        log = []
        complement = function (u, p, t, dt_c)
            push!(log, (; hook = :complement, t, u = copy(u)))
            return nothing
        end
        (; integ) = step_exchange_integrator(
            TrapezoidalSplitOuter(complement);
            log,
            dt,
            fast_dt = dt / 4,
            n_steps = 1,
        )
        CTS.step!(integ)
        complements = findall(e -> e.hook == :complement, log)
        @test length(complements) == 2
        full = findall(e -> e.hook == :cache, log)
        # A full cache refresh occurs before the second complement call, paired
        # with the sub-cycled end state and time.
        refresh = findlast(<(complements[2]), full)
        @test refresh !== nothing
        @test log[full[refresh]].t == dt
        @test log[full[refresh]].u == log[complements[2]].u
    end

    @testset "constrain_state! threading" begin
        for outer in (LieSplitOuter(), TrapezoidalSplitOuter())
            constrain_count = Ref(0)
            constrain! = function (u, p, t)
                constrain_count[] += 1
                u[2] = 0.0
                return nothing
            end
            n_steps = 5
            res = run_step_exchange(
                outer;
                constrain_state! = constrain!,
                dt = 0.1,
                fast_dt = 0.1 / 4,
                n_steps,
            )
            # Applied once per outer step, to the combined end-of-step state.
            @test constrain_count[] == n_steps
            @test res.u[2] == 0.0
        end
    end
end
