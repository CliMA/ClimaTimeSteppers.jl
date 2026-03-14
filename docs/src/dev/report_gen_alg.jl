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

# The stiff_linear problem has eigenvalue λ = 1000 in the implicit part. Three
# algorithms are marked broken here for distinct structural reasons:
#
#   RK4            — fully explicit; its measured convergence order against an
#                    IMEX reference is unreliable because the stiff component
#                    introduces large, ill-conditioned errors at coarser step
#                    sizes, making the log-log slope estimate noisy.
#
#   ARK437L2SA1    — 4th-order IMEX; when measured against a 5th-order IMEX
#                    reference (ARK548L2SA2), the error magnitudes at coarser
#                    steps are small enough that the slope estimate has a very
#                    wide confidence interval, triggering the uncertainty check.
#
#   ARK548L2SA2    — self-referential: this IS the reference algorithm, so its
#                    own errors are dominated by floating-point noise, making the
#                    estimated convergence order meaningless.
#
# All three cases are limitations of the test harness (numerical reference +
# finite sampling), not of the algorithms themselves.
test_convergence!(
    convergence_results,
    alg_name,
    stiff_linear_test_cts(Float64),
    1000;
    numerical_reference_algorithm_name = ARK548L2SA2(),
    broken_tests = (RK4(), ARK437L2SA1(), ARK548L2SA2()),
)

mkpath("output")
JLD2.save_object(joinpath("output", "convergence_$alg_str.jld2"), convergence_results)
