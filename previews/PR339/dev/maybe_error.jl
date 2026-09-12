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

import ClimaTimeSteppers as CTS
parsed_args = parse_commandline()
alg_name = getproperty(CTS, Symbol(parsed_args["alg"]))()

if isfile("output/$alg_name/data_with_nans.dat")
    error("Found NaNs!")
end
