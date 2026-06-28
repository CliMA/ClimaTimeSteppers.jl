import ClimaTimeSteppers as CTS
import ClimaTimeSteppers: solve, init, solve!, step!, reinit!, ODEProblem
using ClimaTimeSteppers, Test

include(joinpath(@__DIR__, "integrator_utils.jl"))
include(joinpath(@__DIR__, "..", "problems.jl"))

@testset "integrator save times" begin
    for (alg, test_case) in
        ((ExplicitAlgorithm(SSP33ShuOsher()), clima_constant_tendency_test(Float64)),),
        reverse_prob in (false, true),
        n_dt_steps in (10, 10000)

        (; prob, analytic_sol) = test_case
        if reverse_prob
            prob = reverse_problem(prob, analytic_sol)
        end

        t0, tf = prob.tspan
        dt = abs(tf - t0) / (n_dt_steps + 0.01) # not aligned with tspan
        save_dt = abs(tf - t0) / 10.02 # not aligned with either tspan or dt

        tdir = tf > t0 ? 1 : -1
        exact_dt_times = accumulate(+, [t0, repeat([tdir * dt], n_dt_steps)...])
        save_dt_times = t0:(tdir * save_dt):tf

        is_just_past_saving_time(t) = any(
            saving_time ->
                tf > t0 ? saving_time <= t < saving_time + dt :
                saving_time - dt < t <= saving_time,
            save_dt_times,
        )
        misaligned_saving_times = filter(is_just_past_saving_time, exact_dt_times)

        adding_callback = get_adding_callback(save_dt_times, t0, tf)
        setting_callback = get_setting_callback()
        n_set_steps = floor(Int, log2(1 + abs(tf - t0) / dt)) - 1
        setting_times = [accumulate(+, [t0, (tdir * dt * 2 .^ (0:n_set_steps))...])..., tf]

        fixed_dt_times = [exact_dt_times..., exact_dt_times[end] + tdir * dt]

        for (compare_to_ode, kwargs, times) in (
            # testing default saving behavior
            (false, (;), [t0, tf]),
            # testing save_everystep
            (true, (; save_everystep = false), [t0, tf]),
            (true, (; save_everystep = true), [exact_dt_times..., tf]),
            (true, (; save_everystep = true, saveat = [t0]), [exact_dt_times..., tf]),
            # testing simple saveat
            (false, (; saveat = []), []),
            (true, (; saveat = [t0]), [t0]),
            (true, (; saveat = [tf]), [tf]),
            # testing non-interpolated saving
            (false, (; saveat = save_dt_times), misaligned_saving_times),
            # testing tstops (tstops remove the need for interpolation when saving)
            (true, (; saveat = save_dt_times, tstops = save_dt_times), save_dt_times),
            # testing add_tstops! and add_saveat!
            (true, (; saveat = [tf], callback = adding_callback), [save_dt_times..., tf]),
            # testing set_dt!
            (false, (; save_everystep = true, callback = setting_callback), setting_times),
            # testing dtchangeable
            (false, (; save_everystep = true, dtchangeable = false), fixed_dt_times),
            (
                false,
                (; save_everystep = true, dtchangeable = false, tstops = save_dt_times),
                fixed_dt_times,
            ),
            # testing stepstop
            (false, (; save_everystep = true, stepstop = 0), exact_dt_times[1:1]),
            (false, (; save_everystep = true, stepstop = 4), exact_dt_times[1:5]),
        )
            is_ode = !(alg isa ClimaTimeSteppers.TimeSteppingAlgorithm)
            is_ode && !compare_to_ode && continue

            sol = solve(deepcopy(prob), alg; dt, kwargs...)

            sol_times = is_ode ? unique(sol.t) : sol.t
            @test sol_times == times

            isempty(times) && continue
            # For constant tendency, RK is exact; error is O(n_dt_steps * eps()).
            # 1e6 * eps() covers the worst case (n_dt_steps = 10000).
            @test sol.u ≈ map(analytic_sol, sol.t) atol = 1000000 * eps()
        end
    end
end

@testset "integrator save times with reinit!" begin
    alg = ExplicitAlgorithm(SSP33ShuOsher())
    test_case = clima_constant_tendency_test(Float64)
    (; prob, analytic_sol) = test_case
    for reverse_prob in (false, true)
        if reverse_prob
            prob = reverse_problem(prob, analytic_sol)
        end

        t0, tf = prob.tspan
        t0′, tf′ = prob.tspan .+ (tf - t0) / 3
        u0′ = analytic_sol(t0′)

        dt = abs(tf - t0) / 100.01 # not aligned with tspan
        save_dt = abs(tf - t0) / 10.02 # not aligned with either tspan or dt

        tdir = tf > t0 ? 1 : -1
        save_dt_times = t0:(tdir * save_dt):tf
        save_dt_times′ = t0′:(tdir * save_dt):tf′
        all_times = [save_dt_times..., tf, save_dt_times′..., tf′]

        for (init_kwargs, reinit_kwargs, times) in (
            ((;), (;), [t0′, tf′]),
            ((;), (; erase_sol = false), [t0, tf, t0′, tf′]),
            ((;), (; saveat = save_dt_times′, tstops = save_dt_times′), save_dt_times′),
            ((; saveat = save_dt_times′, tstops = save_dt_times′), (;), save_dt_times′),
        )
            integrator = init(deepcopy(prob), alg; dt, init_kwargs...)
            @test !@any_reltype(integrator, (UnionAll, DataType))
            solve!(integrator)
            reinit!(integrator, u0′; t0 = t0′, tf = tf′, reinit_kwargs...)
            sol = solve!(integrator)
            @test sol.t == times
            @test sol.u ≈ map(analytic_sol, sol.t) atol = 1000 * eps()
        end
    end
end

@testset "integrator step past end time" begin
    alg = ExplicitAlgorithm(SSP33ShuOsher())
    test_case = clima_constant_tendency_test(Float64)
    (; prob, analytic_sol) = test_case
    t0, tf = prob.tspan
    dt = tf - t0
    integrator = init(deepcopy(prob), alg; dt, save_everystep = true)
    step!(integrator)
    step!(integrator)
    step!(integrator)
    sol = integrator.sol
    @test sol.t == [t0, t0 + dt, t0 + 2 * dt, t0 + 3 * dt]
    @test sol.u ≈ map(analytic_sol, sol.t) atol = 10 * eps()
end

@testset "advance_to_tstop: step! advances to the next tstop" begin
    # With advance_to_tstop = true, a single step! should advance all the way to
    # the next tstop (shortening dt to land on it exactly), not take one base-dt
    # internal step.
    alg = ExplicitAlgorithm(SSP33ShuOsher())
    (; prob, analytic_sol) = clima_constant_tendency_test(Float64)
    tstops = [0.25, 0.7, 1.0]
    integrator =
        init(deepcopy(prob), alg; dt = 0.1, tstops, advance_to_tstop = true)

    step!(integrator)
    @test integrator.t == 0.25            # landed exactly on the first tstop
    @test integrator.u ≈ analytic_sol(0.25) atol = 10 * eps()

    step!(integrator)
    @test integrator.t == 0.7             # not 0.35 (a single base-dt step)
    @test integrator.u ≈ analytic_sol(0.7) atol = 10 * eps()

    step!(integrator)
    @test integrator.t == 1.0
    @test integrator.u ≈ analytic_sol(1.0) atol = 10 * eps()
end

@testset "two-arg step!(integrator, dt, stop_at_tdt)" begin
    alg = ExplicitAlgorithm(SSP33ShuOsher())
    (; prob, analytic_sol) = clima_constant_tendency_test(Float64)

    # stop_at_tdt = true adds t + dt as a tstop and lands on it exactly, even
    # though dt = 0.4 is not a multiple of the base dt = 0.1.
    integrator = init(deepcopy(prob), alg; dt = 0.1)
    step!(integrator, 0.4, true)
    @test integrator.t ≈ 0.4 atol = 1e-12
    @test integrator.u ≈ analytic_sol(integrator.t) atol = 10 * eps()

    t_before = integrator.t
    step!(integrator, 0.25, true)
    @test integrator.t ≈ t_before + 0.25 atol = 1e-12
    @test integrator.u ≈ analytic_sol(integrator.t) atol = 10 * eps()

    # Default stop_at_tdt = false advances to the first step at or past t + dt.
    integrator = init(deepcopy(prob), alg; dt = 0.1)
    step!(integrator, 0.25)
    @test integrator.t ≥ 0.25
    @test integrator.t ≤ 0.25 + 0.1 + 1e-12
    @test integrator.u ≈ analytic_sol(integrator.t) atol = 10 * eps()
end

@testset "integrator API error branches" begin
    alg = ExplicitAlgorithm(SSP33ShuOsher())
    (; prob) = clima_constant_tendency_test(Float64)

    # init: dt must be positive
    @test_throws ErrorException init(deepcopy(prob), alg; dt = -0.1)
    @test_throws ErrorException init(deepcopy(prob), alg; dt = 0.0)

    integrator = init(deepcopy(prob), alg; dt = 0.1)

    # set_dt!: dt must be positive
    @test_throws ErrorException CTS.set_dt!(integrator, -1.0)
    @test_throws ErrorException CTS.set_dt!(integrator, 0.0)

    # two-arg step!: dt must be positive
    @test_throws ErrorException step!(integrator, -0.1)

    # add_tstop! / add_saveat! cannot schedule a time behind the current time
    @test_throws ErrorException CTS.add_tstop!(integrator, -1.0)
    @test_throws ErrorException CTS.add_saveat!(integrator, -1.0)

    # stop_at_tdt = true requires dtchangeable
    fixed = init(deepcopy(prob), alg; dt = 0.1, dtchangeable = false)
    @test_throws ErrorException step!(fixed, 0.1, true)
end

@testset "reinit!(reinit_callbacks = false) leaves user callbacks alone" begin
    # Regression test: with `save = false` there is no saving callback, so the
    # last discrete callback is the user callback. `reinit_callbacks = false`
    # must reinit only the saving callback (here, none) and must NOT
    # re-initialize the user callback.
    alg = ExplicitAlgorithm(SSP33ShuOsher())
    (; prob) = clima_constant_tendency_test(Float64)

    n_inits = Ref(0)
    user_cb = CTS.DiscreteCallback(
        (u, t, integrator) -> false,        # never fires during stepping
        integrator -> nothing;
        initialize = (cb, u, t, integrator) -> (n_inits[] += 1),
    )

    integrator =
        init(deepcopy(prob), alg; dt = 0.1, save = false, callback = user_cb)
    @test n_inits[] == 1   # init runs the callback's initialize once
    step!(integrator)

    # reinit without reinitializing callbacks: user callback must be untouched
    # (and no BoundsError from a missing saving callback)
    reinit!(integrator; reinit_callbacks = false)
    @test n_inits[] == 1

    # reinit_callbacks = true should re-initialize the user callback
    reinit!(integrator; reinit_callbacks = true)
    @test n_inits[] == 2
end
