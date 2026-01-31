module ClimaTimeSteppersBenchmarkToolsExt


import StatsBase: median
import SciMLBase
import PrettyTables
import OrderedCollections
import LinearAlgebra
import ClimaTimeSteppers as CTS
using CUDA
import ClimaComms

import BenchmarkTools

include("benchmark_utils.jl")

# Wrapper that overwrides the default method of ldiv! for W, and also
# - supports the zero function, which is applied to W during SciMLBase.init
# - defines a function that can be applied prior to T_imp!.Wfact
struct LDivOverride{T, C}
    W::T
    custom_ldiv!::C
end
LinearAlgebra.ldiv!(x, A::LDivOverride, b) = A.custom_ldiv!(x, A.W, b)
Base.zero(A::LDivOverride) = LDivOverride(zero(A.W), A.custom_ldiv!)
unwrap_ldiv_override_for_Wfact(A, u, p, dt, t) = (A.W, u, p, dt, t)

"""
    benchmark_step(integrator, device; kwargs...)

Benchmarks one step of a `DistributedODEIntegrator` on a specific `ClimaComms`
device, along with all user-specified functions that are called during the step,
and prints a table that summarizes the measurements. Allocation and timing data
for user-specified functions is rescaled by the number of times they are called
during the step, so that the total costs of different functions can be compared.

Available keyword arguments are
 - `with_cu_prof`: `:profile` or `:bprofile`, to use either `CUDA.@profile` or
   `CUDA.@bprofile` when running on a `CUDADevice`
 - `trace`: boolean flag passed to `CUDA.@profile` (see docs for `CUDA.jl`)
 - `crop`: boolean indicating whether the table printed by `CUDA.@profile`
   should be cropped
 - `hcrop`: number of horizontal characters to include in the cropped table when
   `crop` is `true`
"""
function CTS.benchmark_step(
    integrator::CTS.DistributedODEIntegrator,
    device::ClimaComms.AbstractDevice;
    with_cu_prof = :bprofile,
    trace = false,
    crop = false,
    hcrop = nothing,
    only = nothing, # for backward compatibility
)
    (; u, p, t, dt, alg, cache) = integrator
    (; prob) = integrator.sol
    @assert prob isa SciMLBase.ODEProblem
    @assert prob.f isa CTS.ClimaODEFunction
    (; T_exp_T_lim!, T_exp!, T_lim!, T_imp!, lim!, dss!, cache!, cache_imp!) = prob.f

    if hasproperty(cache, :newtons_method_cache) && !isnothing(cache.newtons_method_cache)
        W = cache.newtons_method_cache.j
    elseif hasproperty(cache, :W)
        W = cache.W
    else
        W = nothing
    end

    n_calls = Dict(
        "ldiv!" => 0,
        "Wfact!" => 0,
        "T_imp!" => 0,
        "T_exp_T_lim!" => 0,
        "T_exp!" => 0,
        "T_lim!" => 0,
        "lim!" => 0,
        "dss!" => 0,
        "cache!" => 0,
        "cache_imp!" => 0,
    )

    # Distinguish the default functions in prob.f from user-specified functions
    is_default(func) = func in (nothing, Returns(nothing))

    # Replicate the logic used in solve_newton! to handle dense matrices
    newton_ldiv!(x, W, b) = LinearAlgebra.ldiv!(x, W isa DenseMatrix ? LinearAlgebra.lu(W) : W, b)

    # Initialize and step an integrator that increments call counts of user-specified functions
    with_count(func, name) = is_default(func) ? func : (args...) -> (func(args...); n_calls[name] += 1)
    ode_func_with_counts = CTS.ClimaODEFunction(;
        T_imp! =
            is_default(T_imp!) ? T_imp! :
            SciMLBase.ODEFunction(
                with_count(T_imp!.f, "T_imp!");
                jac_prototype = LDivOverride(W, with_count(newton_ldiv!, "ldiv!")),
                Wfact = with_count(T_imp!.Wfact ∘ unwrap_ldiv_override_for_Wfact, "Wfact!"),
            ),
        T_exp_T_lim! = with_count(T_exp_T_lim!, "T_exp_T_lim!"),
        T_exp! = with_count(T_exp!, "T_exp!"),
        T_lim! = with_count(T_lim!, "T_lim!"),
        lim! = with_count(lim!, "lim!"),
        dss! = with_count(dss!, "dss!"),
        cache! = with_count(cache!, "cache!"),
        cache_imp! = with_count(cache_imp!, "cache_imp!"),
    )
    prob_with_counts = SciMLBase.ODEProblem(ode_func_with_counts, prob.u0, prob.tspan, prob.p)
    SciMLBase.step!(SciMLBase.init(prob_with_counts, alg; dt))
    n_calls["step!"] = 1

    # Make a list of user-specified functions and arguments with which they can be called
    T_args = (similar(u), u, p, t)
    call_signatures = OrderedCollections.OrderedDict()
    call_signatures["step!"] = (SciMLBase.step!, (integrator,))
    isnothing(W) || (call_signatures["ldiv!"] = (newton_ldiv!, (similar(u), W, u)))
    isnothing(W) || (call_signatures["Wfact!"] = (T_imp!.Wfact, (W, u, p, dt, t)))
    is_default(T_imp!) || (call_signatures["T_imp!"] = (T_imp!, T_args))
    is_default(T_exp_T_lim!) || (call_signatures["T_exp_T_lim!"] = (T_exp_T_lim!, T_args))
    is_default(T_exp!) || (call_signatures["T_exp!"] = (T_exp!, T_args))
    is_default(T_lim!) || (call_signatures["T_lim!"] = (T_lim!, T_args))
    is_default(lim!) || (call_signatures["lim!"] = (lim!, (similar(u), p, t, u)))
    is_default(dss!) || (call_signatures["dss!"] = (dss!, (u, p, t)))
    is_default(cache!) || (call_signatures["cache!"] = (cache!, (u, p, t)))
    is_default(cache_imp!) || (call_signatures["cache_imp!"] = (cache_imp!, (u, p, t)))

    # Run all benchmarks, rescale measured data by call counts, and summarize the results
    trials = OrderedCollections.OrderedDict()
    trial_summaries = OrderedCollections.OrderedDict()
    for (name, (func, args)) in call_signatures
        trial = get_trial(func, args, name, device, with_cu_prof, trace, crop, hcrop)
        rescaled_data = n_calls[name] .* (trial.times, trial.gctimes, trial.memory, trial.allocs)
        trials[name] = BenchmarkTools.Trial(trial.params, rescaled_data...)
        trial_summaries[name] = get_summary(trials[name], trials["step!"])
    end
    print_summary_table(trial_summaries, n_calls)

    return (; table_summary = trial_summaries, trials) # for backward compatibility
end

end
