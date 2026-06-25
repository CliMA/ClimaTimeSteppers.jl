module ClimaTimeSteppersBenchmarkToolsExt

import BenchmarkTools
import BenchmarkTools: median

import LinearAlgebra
import CUDA
import OrderedCollections
import PrettyTables

import ClimaComms
import ClimaTimeSteppers as CTS

include("benchmark_utils.jl")

# Wrapper for W that overwrides the default ldiv! and can be passed to zero()
struct LDivOverride{T, C}
    W::T
    custom_ldiv!::C
end
LinearAlgebra.ldiv!(x, A::LDivOverride, b) = A.custom_ldiv!(x, A.W, b)
Base.zero(A::LDivOverride) = LDivOverride(zero(A.W), A.custom_ldiv!)

"""
    benchmark_step(integrator, [device]; kwargs...)

Benchmarks one step of a `TimeStepperIntegrator` on a specific `ClimaComms`
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
    integrator::CTS.TimeStepperIntegrator,
    device::ClimaComms.AbstractDevice = ClimaComms.device();
    with_cu_prof = :bprofile,
    trace = false,
    crop = false,
    hcrop = nothing,
    only = nothing, # for backward compatibility
)
    (; u, p, t, dt, alg, cache) = integrator
    if hasproperty(cache, :newtons_method_cache) && !isnothing(cache.newtons_method_cache)
        W = cache.newtons_method_cache.j
    elseif hasproperty(cache, :W)
        W = cache.W
    else
        W = nothing
    end

    (; prob) = integrator.sol
    @assert prob isa CTS.ODEProblem && prob.f isa CTS.ClimaODEFunction
    (;
        T_imp!,
        T_exp_T_lim!,
        lim!,
        dss!,
        initialize_imp!,
        cache!,
        cache_imp!,
    ) = prob.f

    call_counts = Dict(
        "ldiv!" => 0,
        "Wfact!" => 0,
        "T_imp!" => 0,
        "T_exp!" => 0,
        "lim!" => 0,
        "dss!" => 0,
        "initialize_imp!" => 0,
        "cache!" => 0,
        "cache_imp!" => 0,
    )

    # Distinguish the default functions in prob.f from user-specified functions
    is_default(f) = f in (nothing, Returns(nothing))

    # Replicate the logic used in solve_newton! to handle dense matrices
    ldiv!(x, W, b) = LinearAlgebra.ldiv!(x, W isa DenseMatrix ? LinearAlgebra.lu(W) : W, b)

    # Initialize and step an integrator that increments function call counts
    with_count(f, name) =
        is_default(f) ? f : (args...) -> (f(args...); call_counts[name] += 1)
    T_imp_with_counts =
        is_default(T_imp!) ? T_imp! :
        CTS.ODEFunction(
            with_count(T_imp!.f, "T_imp!");
            jac_prototype =
            isnothing(W) ? nothing : LDivOverride(W, with_count(ldiv!, "ldiv!")),
            Wfact = with_count((A, args...) -> T_imp!.Wfact(A.W, args...), "Wfact!"),
        )
    ode_func_with_counts = CTS.ClimaODEFunction(;
        T_imp! = T_imp_with_counts,
        T_exp_T_lim! = with_count(T_exp_T_lim!, "T_exp!"),
        lim! = with_count(lim!, "lim!"),
        dss! = with_count(dss!, "dss!"),
        initialize_imp! = with_count(initialize_imp!, "initialize_imp!"),
        cache! = with_count(cache!, "cache!"),
        cache_imp! = with_count(cache_imp!, "cache_imp!"),
    )
    prob_with_counts = CTS.ODEProblem(ode_func_with_counts, prob.u0, prob.tspan, prob.p)
    CTS.step!(CTS.init(prob_with_counts, alg; dt))
    call_counts["step!"] = 1

    # Make a list of all user-defined function calls that occur during each step
    uₜ = similar(u)
    call_signatures = OrderedCollections.OrderedDict()
    call_signatures["step!"] = (CTS.step!, (integrator,))
    isnothing(W) || (call_signatures["ldiv!"] = (ldiv!, (uₜ, W, u)))
    isnothing(W) || (call_signatures["Wfact!"] = (T_imp!.Wfact, (W, u, p, dt, t)))
    is_default(T_imp!) || (call_signatures["T_imp!"] = (T_imp!, (uₜ, u, p, t)))
    is_default(T_exp_T_lim!) ||
        (call_signatures["T_exp!"] = (T_exp_T_lim!, (uₜ, uₜ, u, p, t)))
    is_default(lim!) || (call_signatures["lim!"] = (lim!, (u, p, t, u)))
    is_default(dss!) || (call_signatures["dss!"] = (dss!, (u, p, t)))
    is_default(initialize_imp!) ||
        (call_signatures["initialize_imp!"] = (initialize_imp!, (u, p, t)))
    is_default(cache!) || (call_signatures["cache!"] = (cache!, (u, p, t)))
    is_default(cache_imp!) || (call_signatures["cache_imp!"] = (cache_imp!, (u, p, t)))

    # Benchmark each function, and rescale its measurements by the relevant call count
    trials = OrderedCollections.OrderedDict()
    summaries = OrderedCollections.OrderedDict()
    for (name, (f, args)) in call_signatures
        trials[name] = get_trial(f, args, name, device, with_cu_prof, trace, crop, hcrop)
        summaries[name] = get_summary(trials[name], call_counts[name])
    end

    print_summary_table(summaries, call_counts)

    return (; table_summary = summaries, trials) # for backward compatibility
end

end
