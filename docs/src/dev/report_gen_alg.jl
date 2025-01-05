#=
julia --project=docs
using Revise; include(joinpath("docs", "src", "dev", "report_gen_alg.jl"))

Tested algs:

 - `SSP22Heuns`
 - `SSP33ShuOsher`
 - `RK4`
 - `ARK2GKC`
 - `ARS111`
 - `ARS121`
 - `ARS122`
 - `ARS222`
 - `ARS232`
 - `ARS233`
 - `ARS343`
 - `ARS443`
 - `SSP222`
 - `SSP322`
 - `SSP332`
 - `SSP333`
 - `SSP433`
 - `DBM453`
 - `HOMMEM1`
 - `IMKG232a`
 - `IMKG232b`
 - `IMKG242a`
 - `IMKG242b`
 - `IMKG243a`
 - `IMKG252a`
 - `IMKG252b`
 - `IMKG253a`
 - `IMKG253b`
 - `IMKG254a`
 - `IMKG254b`
 - `IMKG254c`
 - `IMKG342a`
 - `IMKG343a`
 - `SSPKnoth`
=#
using ClimaTimeSteppers
using Test
using InteractiveUtils: subtypes

import ArgParse
function parse_commandline()
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table s begin
        "--alg"
        help = "Algorithm to test convergence"
        arg_type = String
        default = "ARS343"
    end
    parsed_args = ArgParse.parse_args(ARGS, s)
    return parsed_args
end

ENV["GKSwstype"] = "nul" # avoid displaying plots

include(joinpath(@__DIR__, "compute_convergence.jl"))
include(joinpath(pkgdir(ClimaTimeSteppers), "test", "problems.jl"))
import ClimaTimeSteppers as CTS
all_subtypes(::Type{T}) where {T} = isabstracttype(T) ? vcat(all_subtypes.(subtypes(T))...) : [T]
_global_parsed_args = parse_commandline()
_global_alg_name = getproperty(CTS, Symbol(_global_parsed_args["alg"]))()
nans_exported = false
import ClimaCore
ClimaCore.DataLayouts.call_post_op_callback() = true
function ClimaCore.DataLayouts.post_op_callback(data, args...; kwargs...)
    aname = nameof(typeof(_global_alg_name))
    file = "output/$aname/data_with_nans.dat"
    mkpath(dirname(file))
    if any(isnan, parent(data))
        global nans_exported
        if !nans_exported
            @show count(isnan, parent(data))
            @show length(parent(data))
            @show size(parent(data))
            open(file, "w") do io
                println(io, parent(data))
            end
            nans_exported = true
        end
    end
end

let # Convergence
    # NOTE: Some imperfections in the convergence order for SSPKnoth are to be
    # expected because we are not using the exact Jacobian

    parsed_args = parse_commandline()
    alg_name = getproperty(CTS, Symbol(parsed_args["alg"]))()

    export_convergence_results(alg_name, ark_analytic_nonlin_test_cts(Float64), 200)
    export_convergence_results(alg_name, ark_analytic_sys_test_cts(Float64), 400)

    export_convergence_results(alg_name, ark_analytic_test_cts(Float64), 40000; super_convergence = (ARS121(),))
    export_convergence_results(alg_name, onewaycouple_mri_test_cts(Float64), 10000; num_steps_scaling_factor = 5)
    export_convergence_results(
        alg_name,
        climacore_1Dheat_test_cts(Float64),
        400;
        num_steps_scaling_factor = 4,
        numerical_reference_algorithm_name = ARS343(),
    )

    rosenbrock_schemes = filter(name -> name isa ClimaTimeSteppers.RosenbrockAlgorithmName, get_algorithm_names())
    if any(x -> alg_name isa typeof(x), rosenbrock_schemes)
        export_convergence_results(alg_name, climacore_1Dheat_test_implicit_cts(Float64), 400)
    end
    export_convergence_results(
        alg_name,
        climacore_2Dheat_test_cts(Float64),
        600;
        num_steps_scaling_factor = 4,
        numerical_reference_algorithm_name = ARS343(),
    )

    # Unconstrained vs SSP results without limiters
    parsed_args = parse_commandline()
    alg_name = getproperty(CTS, Symbol(parsed_args["alg"]))()
    ssp_schemes = all_subtypes(ClimaTimeSteppers.IMEXSSPRKAlgorithmName)
    if any(x -> alg_name isa x, ssp_schemes)
        test_unconstrained_vs_ssp_without_limiters(alg_name, ark_analytic_nonlin_test_cts(Float64), 200)
        test_unconstrained_vs_ssp_without_limiters(alg_name, ark_analytic_sys_test_cts(Float64), 400)
        test_unconstrained_vs_ssp_without_limiters(alg_name, ark_analytic_test_cts(Float64), 40000)
        test_unconstrained_vs_ssp_without_limiters(alg_name, onewaycouple_mri_test_cts(Float64), 10000)
        test_unconstrained_vs_ssp_without_limiters(alg_name, climacore_1Dheat_test_cts(Float64), 200)
        test_unconstrained_vs_ssp_without_limiters(alg_name, climacore_2Dheat_test_cts(Float64), 200)
    end
end
