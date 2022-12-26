import ClimaTimeSteppers as CTS
import DiffEqBase
using BenchmarkTools, DiffEqBase

include(joinpath(pkgdir(CTS), "test", "problems.jl"))

function main()
    algorithm = CTS.IMEXARKAlgorithm(CTS.ARS343(), CTS.NewtonsMethod(; max_iters = 2))
    dt = 0.01
    for problem in (split_linear_prob_wfact_split(), split_linear_prob_wfact_split_fe())
        integrator = DiffEqBase.init(problem, algorithm; dt)

        cache = CTS.init_cache(problem, algorithm)

        CTS.step_u!(integrator, cache)

        trial = @benchmark CTS.step_u!($integrator, $cache)
        show(stdout, MIME("text/plain"), trial)
    end
end
main()
