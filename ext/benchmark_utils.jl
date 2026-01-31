get_summary(trial, trial_for_step) = (;
    n_samples = length(trial),
    n_allocs = trial.allocs,
    memory = BenchmarkTools.prettymemory(trial.memory),
    t_min = BenchmarkTools.prettytime(minimum(trial.times)),
    t_max = BenchmarkTools.prettytime(maximum(trial.times)),
    t_median = BenchmarkTools.prettytime(median(trial.times)),
    t_percent = median(trial.times) / median(trial_for_step.times) * 100,
)

function print_summary_table(trial_summaries, n_calls)
    names = collect(keys(trial_summaries))
    table_data = hcat(
        names,
        map(name -> trial_summaries[name].n_samples, names),
        map(name -> n_calls[name], names),
        map(name -> trial_summaries[name].n_allocs, names),
        map(name -> trial_summaries[name].memory, names),
        map(name -> trial_summaries[name].t_min, names),
        map(name -> trial_summaries[name].t_max, names),
        map(name -> trial_summaries[name].t_median, names),
        map(name -> round(trial_summaries[name].t_percent; sigdigits = 3), names),
    )
    header = (
        ["Function", "Samples", "Calls", "Memory", "Memory", "Time", "Time", "Time", "Time %"],
        ["", "", "per step", "allocs", "total", "min", "max", "median", "median"],
    )
    alignment = vcat(:l, repeat([:r], length(header[1]) - 1))

    println("NOTE: Measurements are rescaled by the number of calls per step, \
             and may not add up to excactly 100%")
    PrettyTables.pretty_table(table_data; header, alignment, crop = :none)
end

get_trial(f, args, name, device, with_cu_prof, trace, crop, hcrop) =
    if device isa ClimaComms.CUDADevice
        @info "Profiling $name..."
        println()
        p = if with_cu_prof == :bprofile
            CUDA.@bprofile trace = trace f(args...)
        else
            CUDA.@profile trace = trace f(args...)
        end
        if crop
            println(p) # crops by default
        else
            # use "COLUMNS" to set how many horizontal characters to crop:
            # See https://github.com/ronisbr/PrettyTables.jl/issues/11#issuecomment-2145550354
            envs = isnothing(hcrop) ? () : ("COLUMNS" => hcrop,)
            withenv(envs...) do
                io = IOContext(stdout, :crop => :horizontal, :limit => true, :displaysize => displaysize())
                show(io, p)
            end
            println()
        end
        BenchmarkTools.run(BenchmarkTools.@benchmarkable CUDA.@sync $f($(args)...))
    else
        BenchmarkTools.run(BenchmarkTools.@benchmarkable $f($(args)...))
    end
