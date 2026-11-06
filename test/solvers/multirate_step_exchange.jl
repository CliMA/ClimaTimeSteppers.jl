#=
julia --project=test
using Revise; include("test/solvers/multirate_step_exchange.jl")
=#
using ClimaTimeSteppers, LinearAlgebra, Test
import ClimaTimeSteppers as CTS
import LinearAlgebra: mul!, norm

# Linear test problem with two time scales
#
# The operators are combinations of the identity `Id2` and the rotation
# generator `J2`. They commute, so the exact solution is
# `exp((A_exp + A_imp + A_slow) t) Y0`. The three parts are:
# - `A_exp`: fast and explicit, evaluated in the sub-cycle;
# - `A_imp`: fast and implicit, solved by the Newton method of the inner step;
# - `A_slow`: slow, held in the frozen forcing.
# As in a split-explicit model, `freeze!` computes the forcing as the full
# explicit tendency minus the part that the sub-cycle evaluates.
const J2 = [0.0 -1.0; 1.0 0.0]
const Id2 = [1.0 0.0; 0.0 1.0]
const A_exp = 2.0 .* J2
const A_imp = -1.0 .* Id2 .+ 1.0 .* J2
const A_slow = -0.3 .* Id2
const A_full_exp = A_exp .+ A_slow
const A_total = A_exp .+ A_imp .+ A_slow
const Y0 = [1.0, 0.5]

exact_solution(t) = exp(A_total .* t) * Y0

# Build a step-exchange `Multirate` integrator for the test problem.
#
# - `route_lim = true` puts the frozen forcing in the limited part `G_lim`,
#   which the inner `lim!` acts on, instead of in `G`.
# - `log`, when it is a vector, receives `(; hook, t, u)` for each call of
#   `freeze!`, `cache!`, `cache_imp!` and `dss!`, with a copy of the state.
# - The returned counters count the calls of `freeze!` and `lim!`.
function step_exchange_integrator(
    outer; dt, fast_dt, n_steps,
    route_lim = false, log = nothing,
    constrain_state! = Returns(nothing), T_post_imp! = nothing,
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
        T_exp_T_lim! = fast!, T_imp!, T_post_imp!, lim!, constrain_state!,
        cache! = record(:cache), cache_imp! = record(:cache_imp), dss! = record(:dss),
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
                res = run_step_exchange(outer; dt, fast_dt = dt / 8, n_steps = ns)
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
            res = run_step_exchange(outer; dt = 0.1, fast_dt = 0.1 / 4, n_steps)
            @test res.freeze_count == per_step * n_steps
        end
    end

    @testset "limited-forcing path" begin
        args = (; dt = 0.025, fast_dt = 0.025 / 4, n_steps = 20)
        base = run_step_exchange(LieSplitOuter(); route_lim = false, args...)
        limited = run_step_exchange(LieSplitOuter(); route_lim = true, args...)
        @test limited.lim_count > 0
        # Putting the forcing in the limited part gives the same solution.
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
        # Lie calls the complement once per step, over the whole step.
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
        # Trapezoidal calls the complement twice per step, over half a step:
        # at the step start and at the midpoint.
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
            TrapezoidalSplitOuter(); log, dt, fast_dt = dt / 4, n_steps = 1,
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
        # Between the second `freeze!` and the second sub-cycle, `cache!` runs
        # at the step start state and time.
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
            TrapezoidalSplitOuter(complement); log, dt, fast_dt = dt / 4, n_steps = 1,
        )
        CTS.step!(integ)
        complements = findall(e -> e.hook == :complement, log)
        @test length(complements) == 2
        full = findall(e -> e.hook == :cache, log)
        # Before the second complement call, `cache!` runs at the end state of
        # the sub-cycle and at the step end time.
        refresh = findlast(<(complements[2]), full)
        @test refresh !== nothing
        @test log[full[refresh]].t == dt
        @test log[full[refresh]].u == log[complements[2]].u
    end

    @testset "T_post_imp! threading" begin
        for outer in (LieSplitOuter(), TrapezoidalSplitOuter())
            post_count = Ref(0)
            post_imp! = function (du, u, p, t)
                post_count[] += 1
                du .= 0.0
                return nothing
            end
            kwargs = (; dt = 0.1, fast_dt = 0.1 / 4, n_steps = 2)
            res = run_step_exchange(outer; T_post_imp! = post_imp!, kwargs...)
            # `T_post_imp!` runs inside the sub-cycle, and a zero correction
            # leaves the solution unchanged.
            @test post_count[] > 0
            @test res.u == run_step_exchange(outer; kwargs...).u
        end
    end

    @testset "fast_step! sub-step update" begin
        dt = 0.5
        n_sub = 4
        fast_dt = dt / n_sub
        n_steps = 3
        shift = [0.01, -0.03]
        for (Outer, cycles_per_step) in
            ((LieSplitOuter, 1), (TrapezoidalSplitOuter, 2))
            n_updates = cycles_per_step * n_sub * n_steps
            @test Outer(nothing; fast_step! = nothing) == Outer()

            # An update that does nothing leaves the solution unchanged. It adds
            # only its own call and one `cache_imp!` per sub-step to the log.
            ref_log = []
            ref = step_exchange_integrator(Outer(); log = ref_log, dt, fast_dt, n_steps)
            inert_log = []
            inert! = function (u, p, t, sub_dt)
                push!(inert_log, (; hook = :fast_step, t, u = copy(u)))
                return nothing
            end
            inert = step_exchange_integrator(
                Outer(; fast_step! = inert!); log = inert_log, dt, fast_dt, n_steps,
            )
            for _ in 1:n_steps
                CTS.step!(ref.integ)
                CTS.step!(inert.integ)
            end
            @test inert.integ.u == ref.integ.u
            inert_fires = findall(e -> e.hook == :fast_step, inert_log)
            @test length(inert_fires) == n_updates
            @test all(i -> inert_log[i + 1].hook == :cache_imp, inert_fires)
            @test deleteat!(copy(inert_log), sort([inert_fires; inert_fires .+ 1])) ==
                  ref_log

            # An update that changes the state runs once per sub-step of each
            # sub-cycle, after the `dss!` and `cache_imp!` of the inner step.
            # The `cache_imp!` that follows it sees the updated state.
            log = []
            sizes = Float64[]
            update! = function (u, p, t, sub_dt)
                push!(log, (; hook = :fast_step, t, u = copy(u)))
                push!(sizes, sub_dt)
                @. u += shift * sub_dt
                return nothing
            end
            (; integ) = step_exchange_integrator(
                Outer(; fast_step! = update!); log, dt, fast_dt, n_steps,
            )
            n_fires() = count(e -> e.hook == :fast_step, log)
            fires_per_step = map(1:n_steps) do _
                n_before = n_fires()
                CTS.step!(integ)
                return n_fires() - n_before
            end
            @test all(==(cycles_per_step * n_sub), fires_per_step)
            fires = findall(e -> e.hook == :fast_step, log)
            @test length(fires) == n_updates
            @test all(==(fast_dt), sizes)
            @test [log[i].t for i in fires] == [
                (s - 1) * dt + k * fast_dt for s in 1:n_steps for
                _ in 1:cycles_per_step for k in 1:n_sub
            ]
            for i in fires
                @test log[i - 2].hook == :dss
                @test log[i - 1].hook == :cache_imp
                @test log[i - 1].t == log[i].t
                @test log[i - 1].u == log[i].u
                @test log[i + 1].hook == :cache_imp
                @test log[i + 1].t == log[i].t
                @test log[i + 1].u == log[i].u .+ shift .* fast_dt
            end
            @test integ.u != ref.integ.u
        end
    end

    @testset "sub_cycle_start! hook" begin
        dt = 0.5
        n_sub = 4
        fast_dt = dt / n_sub
        n_steps = 3
        for (Outer, cycles_per_step) in
            ((LieSplitOuter, 1), (TrapezoidalSplitOuter, 2))
            @test Outer(nothing; sub_cycle_start! = nothing) == Outer()
            log = []
            start! = function (u, p, t)
                push!(log, (; hook = :sub_cycle_start, t, u = copy(u)))
                return nothing
            end
            (; integ) = step_exchange_integrator(
                Outer(; sub_cycle_start! = start!); log, dt, fast_dt, n_steps,
            )
            ref = step_exchange_integrator(Outer(); dt, fast_dt, n_steps)
            u_starts = map(1:n_steps) do _
                u_start = copy(integ.u)
                CTS.step!(integ)
                CTS.step!(ref.integ)
                return u_start
            end
            starts = findall(e -> e.hook == :sub_cycle_start, log)
            @test length(starts) == cycles_per_step * n_steps
            # Both sub-cycles of a trapezoidal step start from the step start state.
            @test [log[i].t for i in starts] ==
                  [(s - 1) * dt for s in 1:n_steps for _ in 1:cycles_per_step]
            @test [log[i].u for i in starts] ==
                  [u_starts[s] for s in 1:n_steps for _ in 1:cycles_per_step]
            # The hook runs after `freeze!` or `cache!` at the same state, and
            # before the first inner step.
            for i in starts
                @test log[i - 1].hook in (:cache, :freeze)
                @test log[i - 1].u == log[i].u
            end
            # A hook that only records leaves the solution unchanged.
            @test integ.u == ref.integ.u
        end
    end

    @testset "fast_step! with a complement" begin
        dt = 0.5
        n_sub = 4
        fast_dt = dt / n_sub
        n_steps = 2
        for (Outer, cycles_per_step, halves) in
            ((LieSplitOuter, 1, false), (TrapezoidalSplitOuter, 2, true))
            calls = Tuple{Symbol, Float64, Float64}[]
            complement = function (u, p, t, dt_c)
                push!(calls, (:complement, t, dt_c))
                return nothing
            end
            update! = function (u, p, t, sub_dt)
                push!(calls, (:fast_step, t, sub_dt))
                return nothing
            end
            (; integ) = step_exchange_integrator(
                Outer(complement; fast_step! = update!); dt, fast_dt, n_steps,
            )
            for _ in 1:n_steps
                CTS.step!(integ)
            end
            n_updates = cycles_per_step * n_sub
            one_step = [
                :complement
                fill(:fast_step, n_updates)
                halves ? [:complement] : Symbol[]
            ]
            # The complement calls surround the updates of each step. Lie calls
            # it before the sub-cycle, and trapezoidal before and after its two
            # sub-cycles.
            @test first.(calls) == repeat(one_step, n_steps)
            complement_calls = filter(c -> c[1] == :complement, calls)
            @test all(c -> c[3] == (halves ? dt / 2 : dt), complement_calls)
            @test count(c -> c[1] == :fast_step, calls) == n_updates * n_steps
        end
    end

    @testset "fast_step! with a clipped final sub-step" begin
        # With a fast step of 3/16 in a step of 1/2, the final sub-step is 1/8.
        dt = 0.5
        fast_dt = 0.1875
        for (Outer, cycles_per_step) in
            ((LieSplitOuter, 1), (TrapezoidalSplitOuter, 2))
            sizes = Float64[]
            times = Float64[]
            update! = function (u, p, t, sub_dt)
                push!(times, t)
                push!(sizes, sub_dt)
                return nothing
            end
            (; integ) = step_exchange_integrator(
                Outer(; fast_step! = update!); dt, fast_dt, n_steps = 1,
            )
            CTS.step!(integ)
            @test sizes == repeat([fast_dt, fast_dt, 0.125], cycles_per_step)
            @test times == repeat([0.1875, 0.375, 0.5], cycles_per_step)
        end
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
                outer; dt = 0.1, fast_dt = 0.1 / 4, n_steps,
                constrain_state! = constrain!,
            )
            # `constrain_state!` runs once per outer step, on the end state.
            @test constrain_count[] == n_steps
            @test res.u[2] == 0.0
        end
    end
end
