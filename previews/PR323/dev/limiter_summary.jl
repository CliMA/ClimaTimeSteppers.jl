#=
julia --project=docs
using Revise; include(joinpath("docs", "src", "dev", "limiter_summary.jl"))

=#
using ClimaTimeSteppers
using Test

import ClimaTimeSteppers as CTS
include(joinpath(pkgdir(ClimaTimeSteppers), "test", "problems.jl"))
include(joinpath(@__DIR__, "compute_convergence.jl"))
include(joinpath(@__DIR__, "..", "plotting_utils.jl"))
using ClimaTimeSteppers

let # Convergence
    # NOTE: Some imperfections in the convergence order for SSPKnoth are to be
    # expected because we are not using the exact Jacobian
    vector_of_dicts = []
    files = readdir()
    filter!(x -> endswith(x, ".jld2"), files)
    filter!(x -> startswith(basename(x), "limiter_"), files)
    for f in files
        push!(vector_of_dicts, JLD2.load_object(f))
    end

    sol_dicts = merge(vector_of_dicts...)
    algorithm_names = map(x -> x[1], collect(keys(sol_dicts)))
    limiter_summary(sol_dicts, algorithm_names)
end
