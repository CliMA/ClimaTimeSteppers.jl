using ClimaTimeSteppers, Test
import OrdinaryDiffEq

include("problems.jl")

@testset "integrator save times" begin
    test_case = constant_tendency_test
    (; prob, analytic_sol) = test_case
    for alg in (SSPRK33ShuOsher(), OrdinaryDiffEq.SSPRK33()),
        prob in (prob, reverse_problem(prob, analytic_sol)),
        n_dt_steps in (10, 10000)

        t0, tf = prob.tspan
        dt = abs(tf - t0) / (n_dt_steps + 0.01) # not aligned with tspan
        save_dt = abs(tf - t0) / 10.02 # not aligned with either tspan or dt

        tdir = tf > t0 ? 1 : -1
        exact_dt_times = accumulate(+, [t0, repeat([tdir * dt], n_dt_steps)...])
        save_dt_times = t0:(tdir * save_dt):tf

        is_just_past_saving_time(t) = any(
            saving_time -> tf > t0 ? saving_time <= t < saving_time + dt :
                saving_time - dt < t <= saving_time,
            save_dt_times,
        )
        misaligned_saving_times =
            filter(is_just_past_saving_time, exact_dt_times)

        function adding_function!(integrator)
            if integrator.t in save_dt_times
                add_saveat!(integrator, integrator.t)
            end
            next_saving_time_index = findfirst(
                saving_time -> tf > t0 ? saving_time > integrator.t :
                    saving_time < integrator.t,
                save_dt_times,
            )
            isnothing(next_saving_time_index) && return
            add_tstop!(integrator, save_dt_times[next_saving_time_index])
        end
        adding_callback = DiffEqBase.DiscreteCallback(
            (u, t, integrator) -> true,
            adding_function!;
            initialize = (cb, u, t, integrator) -> adding_function!(integrator),
            save_positions = (false, false), # stop OrdinaryDiffEq from saving
        )

        setting_function!(integrator) = ClimaTimeSteppers.set_dt!(
            integrator,
            2 * DiffEqBase.get_dt(integrator),
        )
        setting_callback = DiffEqBase.DiscreteCallback(
            (u, t, integrator) -> true,
            setting_function!,
        )
        n_set_steps = floor(Int, log2(1 + abs(tf - t0) / dt)) - 1
        setting_times =
            [accumulate(+, [t0, (tdir * dt * 2 .^ (0:n_set_steps))...])..., tf]

        fixed_dt_times = [exact_dt_times..., exact_dt_times[end] + tdir * dt]
        
        for (compare_to_ode, kwargs, times) in (
            # testing default saving behavior (OrdinaryDiffEq saves every step by default)
            (false, (;), [t0, tf]),

            # testing save_everystep
            (true, (; save_everystep = false), [t0, tf]),
            (true, (; save_everystep = true), [exact_dt_times..., tf]),
            (true, (; save_everystep = true, saveat = [t0]), [exact_dt_times..., tf]),

            # testing simple saveat (OrdinaryDiffEq saves at [t0, tf] when saveat is empty)
            (false, (; saveat = []), []),
            (true, (; saveat = [t0]), [t0]),
            (true, (; saveat = [tf]), [tf]),

            # testing non-interpolated saving (OrdinaryDiffEq interpolates when saving)
            (false, (; saveat = save_dt_times), misaligned_saving_times),
            (false, (; saveat = save_dt), [misaligned_saving_times..., tf]),

            # testing tstops (tstops remove the need for interpolation when saving)
            (true, (; saveat = save_dt_times, tstops = save_dt_times), save_dt_times),
            (true, (; saveat = save_dt, tstops = save_dt_times), [save_dt_times..., tf]),
            
            # testing add_tstops! and add_saveat!
            (true, (; saveat = [tf], callback = adding_callback), [save_dt_times..., tf]),
            
            # testing set_dt! (OrdinaryDiffEq does not support this function)
            (false, (; save_everystep = true, callback = setting_callback), setting_times),

            # testing dtchangeable (OrdinaryDiffEq does not support this kwarg)
            (false, (; save_everystep = true, dtchangeable = false), fixed_dt_times),
            (false, (; save_everystep = true, dtchangeable = false, tstops = save_dt_times), fixed_dt_times),

            # testing stepstop (OrdinaryDiffEq does not support this kwarg)
            (false, (; save_everystep = true, stepstop = 0), exact_dt_times[1:1]),
            (false, (; save_everystep = true, stepstop = 4), exact_dt_times[1:5]),
        )
            is_ode = !(alg isa ClimaTimeSteppers.DistributedODEAlgorithm)
            is_ode && !compare_to_ode && continue

            # hide the warning about unrecognized kwargs from OrdinaryDiffEq
            kwargshandle = DiffEqBase.KeywordArgSilent

            sol = solve(deepcopy(prob), alg; dt, kwargs..., kwargshandle)

            # remove the duplicate entries put in sol by OrdinaryDiffEq
            sol_times = is_ode ? unique(sol.t) : sol.t
            @test sol_times == times

            isempty(times) && continue
            @test sol.u ≈ map(analytic_sol, sol.t) atol = 10000000 * eps()
            # the atol has to be very large for when n_dt_steps is big
        end
    end
end

@testset "integrator save times with reinit!" begin
    # OrdinaryDiffEq does not save at t0′ after reinit! unless erase_sol is
    # true, so this test does not include a comparison with OrdinaryDiffEq.
    alg = SSPRK33ShuOsher()
    test_case = constant_tendency_test
    (; prob, analytic_sol) = test_case
    for prob in (prob, reverse_problem(prob, analytic_sol))
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
            ((;), (; saveat = save_dt, tstops = save_dt_times′), [save_dt_times′..., tf′]),
            ((; saveat = save_dt, tstops = save_dt_times′), (;), [save_dt_times′..., tf′]),
            ((; saveat = save_dt, tstops = all_times), (; erase_sol = false), all_times),
        )
            integrator = init(deepcopy(prob), alg; dt, init_kwargs...)
            solve!(integrator)
            reinit!(integrator, u0′; t0 = t0′, tf = tf′, reinit_kwargs...)
            sol = solve!(integrator)
            @test sol.t == times
            @test sol.u ≈ map(analytic_sol, sol.t) atol = 1000 * eps()
        end
    end
end

@testset "integrator step past end time" begin
    alg = SSPRK33ShuOsher()
    test_case = constant_tendency_test
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

@testset "integrator progress callbacks" begin
    alg = SSPRK33ShuOsher()
    test_case = constant_tendency_test
    (; prob, analytic_sol) = test_case
    t0, tf = prob.tspan
    dt = (tf - t0) / 8 # use 8 (power of 2) steps to avoid round-off errors

    callback = DiffEqBase.DiscreteCallback(
        (u, t, integrator) -> true,
        integrator -> sleep(integrator.step == 1 ? 2 : 0.5),
    ) # make step 1 take 2 seconds and steps 2--8 each take 0.5 seconds

    # hide the warning about unrecognized kwargs from OrdinaryDiffEq
    kwargshandle = DiffEqBase.KeywordArgSilent

    test_io = IOBuffer() # buffer for testing the outputs of progress callbacks

    for io in (stdout, test_io), # run once as a demo and once as a unit test
        prob in (prob, reverse_problem(prob, analytic_sol))
        progress = :basic
        progress_kwargs = (; io, bar_length = 70)
        kwargs = (; dt, callback, progress, progress_kwargs, kwargshandle)
        solve(deepcopy(prob), alg; kwargs...)
        io == test_io && @test String(take!(io)) == """
                      0     10     20     30     40     50     60     70     80     90    100   (%)
                      ▌     ▐▌     ▐▌     ▐▌     ▐▌     ▐▌     ▐▌     ▐▌     ▐▌     ▐▌     ▐
            Progress: ██████████████████████████████████████████████████████████████████████
            """

        progress = :terminal
        custom_message = (integrator, terminal_width) ->
            if integrator.t == prob.tspan[1]
                "Starting\nAnother Line To Print"
            elseif integrator.t == prob.tspan[2]
                "Finished\nAnother Line To Print"
            else
                "Terminal Width: $terminal_width"
            end
        progress_kwargs = (; io, custom_message)
        kwargs = (; dt, callback, progress, progress_kwargs, kwargshandle)
        solve(deepcopy(prob), alg; kwargs...)
        clear_line = "\e[1A\r\e[0K"
        io == test_io && @test String(take!(io)) == """
            Progress: ▐                                                        ▌ 0.0%
            Time Remaining: 
            Starting
            Another Line To Print
            $(clear_line^4)Progress: ▐███████                                                 ▌ 12.5%
            Time Remaining: ...
            Terminal Width: 80
            $(clear_line^3)Progress: ▐██████████████                                          ▌ 25.0%
            Time Remaining: 3 seconds
            Terminal Width: 80
            $(clear_line^3)Progress: ▐█████████████████████                                   ▌ 37.5%
            Time Remaining: 3 seconds
            Terminal Width: 80
            $(clear_line^3)Progress: ▐████████████████████████████                            ▌ 50.0%
            Time Remaining: 2 seconds
            Terminal Width: 80
            $(clear_line^3)Progress: ▐███████████████████████████████████                     ▌ 62.5%
            Time Remaining: 2 seconds
            Terminal Width: 80
            $(clear_line^3)Progress: ▐██████████████████████████████████████████              ▌ 75.0%
            Time Remaining: 1 second
            Terminal Width: 80
            $(clear_line^3)Progress: ▐█████████████████████████████████████████████████       ▌ 87.5%
            Time Remaining: 1 second
            Terminal Width: 80
            $(clear_line^3)Progress: ▐████████████████████████████████████████████████████████▌ 100.0%
            Time Remaining: 0 seconds
            Finished
            Another Line To Print
            $(clear_line^4)"""
        # the "width" of an IOBuffer (the value returned by displaysize) is 80
    end

    close(test_io)
end
