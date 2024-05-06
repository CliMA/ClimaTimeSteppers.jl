module ClimaTimeSteppersBenchmarkToolsExt


import StatsBase
import SciMLBase
import PrettyTables
import OrderedCollections
import LinearAlgebra as LA
import ClimaTimeSteppers as CTS
using CUDA
import ClimaComms

import BenchmarkTools
Base.:*(t::BenchmarkTools.Trial, n::Int) = n * t
Base.:*(n::Int, t::BenchmarkTools.Trial) =
    BenchmarkTools.Trial(t.params, t.times .* n, t.gctimes .* n, t.memory * n, t.allocs * n)

include("benchmark_utils.jl")

function get_n_calls_per_step(integrator::CTS.DistributedODEIntegrator)
    (; alg) = integrator
    (; newtons_method, name) = alg
    (; max_iters) = newtons_method
    n_calls_per_step(name, max_iters)
end

# TODO: generalize
n_calls_per_step(::CTS.ARS343, max_newton_iters) = Dict(
    "Wfact" => 3 * max_newton_iters,
    "ldiv!" => 3 * max_newton_iters,
    "T_imp!" => 3 * max_newton_iters,
    "T_exp_T_lim!" => 4,
    "lim!" => 4,
    "dss!" => 4,
    "post_explicit!" => 3,
    "post_implicit!" => 4,
    "step!" => 1,
)


"""
    benchmark_step(
        integrator::DistributedODEIntegrator,
        device::ClimaComms.AbstractDevice;
        with_cu_prof = :bfrofile, # [:bprofile, :profile]
        trace = false
    )

Benchmark a DistributedODEIntegrator
"""
function CTS.benchmark_step(
    integrator::CTS.DistributedODEIntegrator,
    device::ClimaComms.AbstractDevice;
    with_cu_prof = :bprofile,
    trace = false,
)
    (; u, p, t, dt, sol, alg) = integrator
    (; f) = sol.prob
    if f isa CTS.ClimaODEFunction

        W = get_W(integrator)
        X = similar(u)
        @. X = u
        trials₀ = OrderedCollections.OrderedDict()

#! format: off
		trials₀["Wfact"]          = get_trial(wfact_fun(integrator), (W, u, p, dt, t), "Wfact", device; with_cu_prof, trace);
		trials₀["ldiv!"]          = get_trial(LA.ldiv!, (X, W, u), "ldiv!", device; with_cu_prof, trace);
		trials₀["T_imp!"]         = get_trial(implicit_fun(integrator), implicit_args(integrator), "T_imp!", device; with_cu_prof, trace);
        trials₀["T_exp_T_lim!"]   = get_trial(remaining_fun(integrator), remaining_args(integrator), "T_exp_T_lim!", device; with_cu_prof, trace);
        trials₀["lim!"]           = get_trial(f.lim!, (X, p, t, u), "lim!", device; with_cu_prof, trace);
		trials₀["dss!"]           = get_trial(f.dss!, (u, p, t), "dss!", device; with_cu_prof, trace);
        trials₀["post_explicit!"] = get_trial(f.post_explicit!, (u, p, t), "post_explicit!", device; with_cu_prof, trace);
        trials₀["post_implicit!"] = get_trial(f.post_implicit!, (u, p, t), "post_implicit!", device; with_cu_prof, trace);
		trials₀["step!"]          = get_trial(SciMLBase.step!, (integrator, ), "step!", device; with_cu_prof, trace);
#! format: on

        trials = OrderedCollections.OrderedDict()

        n_calls_per_step = get_n_calls_per_step(integrator)
        for k in keys(trials₀)
            isnothing(trials₀[k]) && continue
            trials[k] = trials₀[k] * n_calls_per_step[k]
        end

        table_summary = OrderedCollections.OrderedDict()
        for k in keys(trials)
            isnothing(trials[k]) && continue
            table_summary[k] = get_summary(trials[k], trials["step!"])
        end

        tabulate_summary(table_summary; n_calls_per_step)

        return (; table_summary, trials)
    else
        @warn "`ClimaTimeSteppers.benchmark` is not yet supported for $f."
        return (; table_summary = nothing, trials = nothing)
    end
end


end
