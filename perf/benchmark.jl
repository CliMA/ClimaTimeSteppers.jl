import ClimaTimeSteppers as CTS
import DiffEqBase
using BenchmarkTools, DiffEqBase

using CUDA, BenchmarkTools, OrderedCollections, StatsBase, PrettyTables # needed for CTS.benchmark_step

include(joinpath(pkgdir(CTS), "test", "problems.jl"))

function main()
    algorithm = CTS.IMEXAlgorithm(CTS.ARS343(), CTS.NewtonsMethod(; max_iters = 2))
    dt = 0.01
    device = ClimaComms.device()
    for problem in (split_linear_prob_wfact_split(), split_linear_prob_wfact_split_fe())
        integrator = DiffEqBase.init(problem, algorithm; dt)

        cache = CTS.init_cache(problem, algorithm)

        CTS.benchmark_step(integrator, device)
    end
end
main()
