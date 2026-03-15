import ArgParse
import JLD2

include(joinpath(@__DIR__, "compute_convergence.jl"))
include(joinpath(pkgdir(ClimaTimeSteppers), "test", "problems.jl"))

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
parsed_args = parse_commandline()
alg_str = parsed_args["alg"]
alg_name = getproperty(ClimaTimeSteppers, Symbol(alg_str))()

convergence_results = Dict()

test_convergence!(convergence_results, alg_name, ark_analytic_nonlin_test_cts(Float64), 450)
test_convergence!(convergence_results, alg_name, ark_analytic_sys_test_cts(Float64), 200)
test_convergence!(
    convergence_results,
    alg_name,
    ark_analytic_test_cts(Float64),
    25000;
    super_convergence_algorithm_names = (ARS121(),),
)
test_convergence!(
    convergence_results,
    alg_name,
    onewaycouple_mri_test_cts(Float64),
    9000;
    high_order_sample_shifts = 3,
)

test_convergence!(
    convergence_results,
    alg_name,
    finitediff_1Dheat_test_cts(Float64),
    200;
    numerical_reference_algorithm_name = ARK548L2SA2(),
)

test_convergence!(
    convergence_results,
    alg_name,
    finitediff_2Dheat_test_cts(Float64),
    200;
    numerical_reference_algorithm_name = ARK548L2SA2(),
)

test_convergence!(
    convergence_results,
    alg_name,
    stiff_linear_test_cts(Float64),
    1000;
    numerical_reference_algorithm_name = ARK548L2SA2(),
)

mkpath("output")
JLD2.save_object(joinpath("output", "convergence_$alg_str.jld2"), convergence_results)
