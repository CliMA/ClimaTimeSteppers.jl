"""
    @any_reltype(::Any, t::Tuple, warn=true)

Returns a Bool (and prints warnings) if the given
data structure has an instance of any types in `t`.
"""
function any_reltype(found, obj, name, ets, pc = (); warn = true)
    for pn in propertynames(obj)
        prop = if obj isa Base.Pairs
            values(obj)
        else
            getproperty(obj, pn)
        end
        pc_full = (pc..., ".", pn)
        pc_string = name * string(join(pc_full))
        for et in ets
            if prop isa et
                warn && @warn "$pc_string::$(typeof(prop)) is a DataType"
                found = true
            end
        end
        found = found || any_reltype(found, prop, name, ets, pc_full; warn)
    end
    return found
end
macro any_reltype(obj, ets, warn = true)
    return :(any_reltype(
        false,
        $(esc(obj)),
        $(string(obj)),
        $(esc(ets));
        warn = $(esc(warn)),
    ))
end

function get_adding_callback(save_dt_times, t0, tf)
    function adding_function!(integrator)
        if integrator.t in save_dt_times
            ClimaTimeSteppers.add_saveat!(integrator, integrator.t)
        end
        next_saving_time_index = findfirst(
            saving_time ->
                tf > t0 ? saving_time > integrator.t : saving_time < integrator.t,
            save_dt_times,
        )
        isnothing(next_saving_time_index) && return
        ClimaTimeSteppers.add_tstop!(integrator, save_dt_times[next_saving_time_index])
    end
    return ClimaTimeSteppers.DiscreteCallback(
        (u, t, integrator) -> true,
        adding_function!;
        initialize = (cb, u, t, integrator) -> adding_function!(integrator),
    )
end

function get_setting_callback()
    setting_function!(integrator) =
        ClimaTimeSteppers.set_dt!(integrator, 2 * ClimaTimeSteppers.get_dt(integrator))
    return ClimaTimeSteppers.DiscreteCallback((u, t, integrator) -> true, setting_function!)
end
