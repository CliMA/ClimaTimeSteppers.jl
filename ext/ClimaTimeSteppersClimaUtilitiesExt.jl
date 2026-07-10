module ClimaTimeSteppersClimaUtilitiesExt

import ClimaTimeSteppers as CTS
import ClimaUtilities.TimeManager: ITime
import Dates

# `ITime` methods for `CTS.sub_timestep` and `CTS.refine_time`: divide and refine
# in the nanosecond period.
function CTS.sub_timestep(dt::ITime, n_sub::Integer)
    dt_ns = dt.counter * Dates.tons(dt.period)
    q, r = divrem(dt_ns, n_sub)
    iszero(r) || throw(
        ArgumentError(
            "dt = $dt is not divisible by n_sub = $n_sub at nanosecond resolution",
        ),
    )
    return ITime(q; period = Dates.Nanosecond(1), epoch = dt.epoch)
end

CTS.refine_time(t::ITime) = ITime(
    t.counter * Dates.tons(t.period);
    period = Dates.Nanosecond(1),
    epoch = t.epoch,
)

end
