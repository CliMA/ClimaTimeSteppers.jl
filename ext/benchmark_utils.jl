get_summary(trial, call_count) = (;
    allocs = trial.allocs * call_count,
    memory = trial.memory * call_count,
    time_min = minimum(trial.times) * call_count,
    time_max = maximum(trial.times) * call_count,
    time_median = median(trial.times) * call_count,
)

rounded_percent(partial, total) = round(partial / total * 100; digits = 2)

function print_summary_table(summaries, call_counts)
    names = collect(keys(summaries))
    step_allocs = summaries["step!"].allocs
    step_memory = summaries["step!"].memory
    step_time = summaries["step!"].time_median
    other_allocs = 2 * step_allocs - sum(name -> summaries[name].allocs, names)
    other_memory = 2 * step_memory - sum(name -> summaries[name].memory, names)
    other_time = 2 * step_time - sum(name -> summaries[name].time_median, names)
    other_memory_str = BenchmarkTools.prettymemory(other_memory)
    other_time_str = BenchmarkTools.prettytime(other_time)
    other_percent = rounded_percent(other_time, step_time)
    table_data = vcat(
        hcat(
            names,
            map(name -> call_counts[name], names),
            map(name -> summaries[name].allocs, names),
            map(name -> BenchmarkTools.prettymemory(summaries[name].memory), names),
            map(name -> BenchmarkTools.prettytime(summaries[name].time_min), names),
            map(name -> BenchmarkTools.prettytime(summaries[name].time_max), names),
            map(name -> BenchmarkTools.prettytime(summaries[name].time_median), names),
            map(name -> rounded_percent(summaries[name].time_median, step_time), names),
        ),
        ["other" "--" other_allocs other_memory_str "--" "--" other_time_str other_percent],
    )
    column_labels = [
        ["", "Calls", "Memory", "Memory", "Time", "Time", "Time", "Step %"],
        ["", "per step", "allocs", "total", "min", "max", "median", "median"],
    ]
    alignment = vcat(:l, repeat([:r], length(column_labels[1]) - 1))
    PrettyTables.pretty_table(table_data; column_labels, alignment)
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
                io = IOContext(
                    stdout,
                    :crop => :horizontal,
                    :limit => true,
                    :displaysize => displaysize(),
                )
                show(io, p)
            end
            println()
        end
        BenchmarkTools.run(BenchmarkTools.@benchmarkable CUDA.@sync $f($(args)...))
    else
        BenchmarkTools.run(BenchmarkTools.@benchmarkable $f($(args)...))
    end
