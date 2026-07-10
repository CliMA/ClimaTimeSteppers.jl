module ClimaTimeSteppersClimaUtilitiesExt

import ClimaTimeSteppers as CTS
import ClimaUtilities.TimeManager: ITime
import Dates

# Exact sub-step division and nanosecond refinement for `ITime`, extending the
# generic identity methods in ClimaTimeSteppers. The inner integrator of a
# step-exchange `Multirate` runs in the nanosecond period of its sub-step, so
# outer times are refined to that period before assignment, and dividing the
# outer step stays exact even when the outer period is coarse relative to
# `n_sub`.
function CTS.sub_timestep(dt::ITime, n_sub::Integer)
    dt_ns = dt.counter * Dates.tons(dt.period)
    return ITime(div(dt_ns, n_sub); period = Dates.Nanosecond(1), epoch = dt.epoch)
end

CTS.refine_ns(t::ITime) = ITime(
    t.counter * Dates.tons(t.period);
    period = Dates.Nanosecond(1),
    epoch = t.epoch,
)

end
