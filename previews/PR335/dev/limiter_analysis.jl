#=
julia --project=docs
using Revise; include(joinpath("docs", "src", "dev", "limiter_analysis.jl"))

Tested algs:

 - `--alg SSP333 --use_limiter true --use_hyperdiffusion true`
 - `--alg ARS343 --use_limiter true --use_hyperdiffusion true`
 - `--alg SSP333 --use_limiter false --use_hyperdiffusion true`
 - `--alg ARS343 --use_limiter false --use_hyperdiffusion true`
 - `--alg SSP333 --use_limiter true --use_hyperdiffusion false`
 - `--alg ARS343 --use_limiter true --use_hyperdiffusion false`
 - `--alg SSP333 --use_limiter false --use_hyperdiffusion false`
 - `--alg ARS343 --use_limiter false --use_hyperdiffusion false`
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
        "--use_limiter"
        help = "Bool indicating to use limiter"
        arg_type = Bool
        default = true
        "--use_hyperdiffusion"
        help = "Bool indicating to use hyperdiffusion"
        arg_type = Bool
        default = true
    end
    parsed_args = ArgParse.parse_args(ARGS, s)
    return parsed_args
end

ENV["GKSwstype"] = "nul" # avoid displaying plots

import ClimaTimeSteppers as CTS
include(joinpath(pkgdir(ClimaTimeSteppers), "test", "problems.jl"))
using ClimaTimeSteppers
import JLD2

parsed_args = parse_commandline()
alg_name = getproperty(CTS, Symbol(parsed_args["alg"]))()
use_limiter = parsed_args["use_limiter"]
use_hyperdiffusion = parsed_args["use_hyperdiffusion"]


# This also runs with num_steps = 1000, but with larger under/overshoots; 4800
# is the value used in the paper.

FT = Float64
num_steps = 4800
out_dict = Dict()

test_case = horizontal_deformational_flow_test(FT; use_limiter, use_hyperdiffusion)
prob = test_case.split_prob
dt = test_case.t_end / num_steps
algorithm =
    alg_name isa ClimaTimeSteppers.IMEXARKAlgorithmName ? IMEXAlgorithm(alg_name, NewtonsMethod()) :
    ExplicitAlgorithm(alg_name)
solution = solve(deepcopy(prob), algorithm; dt).u
out_dict[alg_name, use_hyperdiffusion, use_limiter] = solution

JLD2.save_object("limiter_$(alg_name)_hyperdiff_$(use_hyperdiffusion)_lim_$(use_limiter).jld2", out_dict)
