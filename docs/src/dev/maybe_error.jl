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
aname = nameof(typeof(alg_name))

file = "output/$aname/data_with_nans.dat"
@assert isdir(dirname(file))
if isfile(file)
    error("Found NaNs!")
end
