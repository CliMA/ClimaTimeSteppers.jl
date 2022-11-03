export BasicProgressCallback, TerminalProgressCallback

# '█' = \blockfull
# '▐' = \blockrighthalf
# '▌' = \blocklefthalf

"""
    BasicProgressCallback(; kwargs...)

Creates a `DiffEqBase.DiscreteCallback` that prints out a progress bar as the
integrator runs.

# Keywords
- `io::IO = stdout`: output stream to which the progress bar is printed
- `min_update_time::Real = 0.5`: minimum delay (in seconds) between prints
- `bar_title::String = "Progress: "`: string that gets printed to the left of
    the progress bar
- `bar_length::Integer = 100`: number of characters in the progress bar
- `show_tens_intervals::Bool = true`: whether to add ticks above the progress
    bar that indicate the positions of 0%, 10%, 20%, ..., 100%; if this is
    `false`, only the ticks at 0% and 100% are shown
"""
function BasicProgressCallback(;
    io = stdout,
    min_update_time = 0.5,
    bar_title = "Progress: ",
    bar_length = 100,
    show_tens_intervals = true,
)
    if show_tens_intervals
        if bar_length % 10 != 0
            new_bar_length = bar_length - bar_length % 10
            @warn "bar_length must be a multiple of 10 if show_tens_intervals \
                   is true; decreasing the given value of $bar_length to \
                   $new_bar_length"
            bar_length = new_bar_length
        end
        interval_length = bar_length ÷ 10
        interval_labels =
            prod(i -> '0' * lpad(i, interval_length - 1), 1:10) * "0   (%)"
        interval_ticks = ('▌' * ' '^(interval_length - 2) * '▐')^10
        bar_header =
            ' '^(length(bar_title)) * interval_labels * '\n' *
            ' '^(length(bar_title)) * interval_ticks * '\n' * bar_title
    else
        edge_ticks = '▌' * ' '^(bar_length - 2) * '▐'
        bar_header = ' '^(length(bar_title)) * edge_ticks * '\n' * bar_title
    end
    time = Ref{Float64}()
    prev_time = Ref{Float64}()
    prev_filled_bar_length = Ref{Int}()
    function initialize(cb, u, t, integrator)
        print(io, bar_header)
        prev_time[] = time_ns() / 1e9
        prev_filled_bar_length[] = 0
    end
    function condition(u, t, integrator)
        time[] = time_ns() / 1e9
        return time[] >= prev_time[] + min_update_time ||
               isempty(integrator.tstops)
    end
    function affect!(integrator)
        t0 = integrator.sol.prob.tspan[1]
        tf = maximum(integrator.tstops.valtree)
        bar_fill_ratio = (integrator.t - t0) / (tf - t0)
        filled_bar_length = floor(Int, bar_length * bar_fill_ratio)
        added_bar_length = filled_bar_length - prev_filled_bar_length[]
        added_bar_length > 0 && print(io, '█'^added_bar_length)
        prev_time[] = time[]
        prev_filled_bar_length[] = filled_bar_length
    end
    function finalize(cb, u, t, integrator)
        added_bar_length = bar_length - prev_filled_bar_length[]
        println(io, '█'^added_bar_length)
    end
    return DiffEqBase.DiscreteCallback(condition, affect!; initialize, finalize)
end

"""
    TerminalProgressCallback([tType]; kwargs...)

Creates a `DiffEqBase.DiscreteCallback` that prints out a progress bar as the
integrator runs, along with an estimate of the time remaining until the
integrator is finished, and also an optional user-specified message (e.g., 
diagnostic information about the integrator's state, or a Unicode plot).

This progress bar is designed to be overwritten on every update, so it should
only be printed to a UNIX terminal (i.e., a terminal that supports clearing the
previous line by printing the control sequence `"\\e[1A\\r\\e[0K"`).

The time remaining is estimated by computing an exponential moving average of
the integrator's speed (seconds of real time that elapse per unit of integrator
time) and multiplying this average speed by the remaining integrator time; the
first integrator step is not included in the average so as to avoid biasing the
estimate with compilation time.

# Arguments
- `tType::Type = Float64`: type of `integrator.t`

# Keywords
- `io::IO = stdout`: terminal output stream to which the progress bar is printed
- `min_update_time::Real = 0.5`: minimum delay (in seconds) between prints
- `bar_title::String = "Progress: "`: string that gets printed to the left of
    the progress bar
- `relative_bar_length::Real = 0.7`: ratio between the number of characters in
    the progress bar and the width of the terminal (`displaysize(io)[2]`)
- `eta_title::String = "Time Remaining: "`: string that gets printed to the left
    of the time remaining
- `new_speed_weight::Real = 0.1`: weight of the new speed in the formula for
    updating the exponential moving average —
    `average := new_speed_weight * new_speed + (1 - new_speed_weight) * average`
- `custom_message::Union{Nothing, Function} = nothing`: an optional function of
    the form `(integrator, terminal_width) -> String` that generates a custom
    message whenever the progress bar is updated; the generated string can have
    multiple lines, and it is recommended to limit the length of each line to be
    no bigger than the terminal width in order to improve readability
- `clear_when_finished::Bool = true`: whether to clear away the progress bar
    (and any other information that was printed) when the integrator is finished
"""
function TerminalProgressCallback(
    ::Type{tType} = Float64;
    io = stdout,
    min_update_time = 0.5,
    bar_title = "Progress: ",
    relative_bar_length = 0.7,
    eta_title = "Time Remaining: ",
    new_speed_weight = 0.1,
    custom_message = nothing,
    clear_when_finished = true,
) where {tType}
    clear_line = "\e[1A\r\e[0K"
    time = Ref{Float64}()
    prev_time = Ref{Float64}()
    prev_progress_string = Ref{String}()
    is_first_step = Ref{Bool}()
    is_first_speed = Ref{Bool}()
    prev_t = Ref{tType}()
    average_speed = Ref{typeof(one(Float64) / one(tType))}()
    function clear_prev_progress_string(terminal_width)
        prev_progress_string_height = sum(
            line -> max(1, cld(length(line), terminal_width)),
            eachsplit(prev_progress_string[], '\n'),
        )
        return clear_line^prev_progress_string_height
    end
    function initialize(cb, u, t, integrator)
        terminal_width = displaysize(io)[2]
        bar_length = floor(Int, terminal_width * relative_bar_length)
        progress_string =
            bar_title * '▐' * ' '^bar_length * "▌ 0.0%\n" * eta_title
        if !isnothing(custom_message)
            progress_string *= '\n' * custom_message(integrator, terminal_width)
        end
        println(io, progress_string)
        prev_time[] = time_ns() / 1e9
        prev_progress_string[] = progress_string
        is_first_step[] = true
    end
    function condition(u, t, integrator)
        time[] = time_ns() / 1e9
        return time[] >= prev_time[] + min_update_time ||
               isempty(integrator.tstops)
    end
    function affect!(integrator)
        terminal_width = displaysize(io)[2]
        clear_string = clear_prev_progress_string(terminal_width)
        bar_length = floor(Int, terminal_width * relative_bar_length)
        t0 = integrator.sol.prob.tspan[1]
        tf = maximum(integrator.tstops.valtree)
        bar_fill_ratio = (integrator.t - t0) / (tf - t0)
        filled_bar_length = floor(Int, bar_length * bar_fill_ratio)
        percent_string = ' ' * string(floor(1000 * bar_fill_ratio) / 10) * '%'
        if is_first_step[]
            eta_string = "..."
            is_first_step[] = false
            is_first_speed[] = true
        else
            new_speed = (time[] - prev_time[]) / (integrator.t - prev_t[])
            if is_first_speed[]
                average_speed[] = new_speed
                is_first_speed[] = false
            else
                average_speed[] =
                    new_speed_weight * new_speed +
                    (1 - new_speed_weight) * average_speed[]
            end
            eta = round(Int, average_speed[] * (tf - integrator.t))
            eta_string = eta == 0 ? "0 seconds" :
                string(Dates.canonicalize(Dates.Second(eta)))
        end
        progress_string =
            bar_title * '▐' * '█'^filled_bar_length *
            ' '^(bar_length - filled_bar_length) * '▌' * percent_string * '\n' *
            eta_title * eta_string
        if !isnothing(custom_message)
            progress_string *= '\n' * custom_message(integrator, terminal_width)
        end
        println(io, clear_string * progress_string)
        prev_time[] = time[]
        prev_progress_string[] = progress_string
        prev_t[] = integrator.t
    end
    function finalize(cb, u, t, integrator)
        terminal_width = displaysize(io)[2]
        clear_string = clear_prev_progress_string(terminal_width)
        if clear_when_finished
            print(io, clear_string)
        else
            bar_length = floor(Int, terminal_width * relative_bar_length)
            progress_string *=
                bar_title * '▐' * '█'^bar_length * "▌ 100.0%\n" * eta_title
            if !isnothing(custom_message)
                progress_string *=
                    '\n' * custom_message(integrator, terminal_width)
            end
            println(io, clear_string * progress_string)
        end
    end
    return DiffEqBase.DiscreteCallback(condition, affect!; initialize, finalize)
end
