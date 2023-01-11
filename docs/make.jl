using Documenter, DocumenterCitations
using ClimaTimeSteppers

# https://github.com/jheinen/GR.jl/issues/278#issuecomment-587090846
ENV["GKSwstype"] = "nul"

bib = CitationBibliography(joinpath(@__DIR__, "refs.bib"))

#! format: off
pages = [
    # "index.md",
    # "Algorithm formulations" => [
    #     "algo_formulations/lsrk.md",
    #     "algo_formulations/ssprk.md",
    #     "algo_formulations/ark.md",
    #     "algo_formulations/mrrk.md",
    # ],
    # "Non-linear solvers" => [
    #     "Formulation" => "nl_solvers/formulation.md",
    #     "Newtons method" => "nl_solvers/newtons_method.md",
    # ],
    "Test problems" => [
        "test_problems/index.md",
        # "test_problems/diffusion_2d.md",
        "test_problems/diffusion_2d_dimensional.md",
    ],
    # "API docs" => [
    #     "Algorithms" => "api/algorithms.md",
    #     "Tableaus" => "api/tableaus.md",
    #     "Non-linear solvers" => "api/nl_solvers.md",
    #     "Callbacks" => "api/callbacks.md",
    # ],
    # # "Algorithm comparisons" => "algo_comparisons.md", # TODO: fill out
    # "Developer docs" => [
    #     "Types" => "dev/types.md",
    #     "Report generator" => "dev/report_gen.md",
    # ],
    # "references.md",
]
#! format: on

mathengine = MathJax(Dict(:TeX => Dict(:equationNumbers => Dict(:autoNumber => "AMS"), :Macros => Dict())))

format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true", mathengine = mathengine, collapselevel = 1)

makedocs(
    bib,
    sitename = "ClimaTimeSteppers",
    format = format,
    modules = [ClimaTimeSteppers],
    checkdocs = :exports,
    clean = true,
    strict = true,
    pages = pages,
    # draft = true, # , skips expensive parts of makedocs, for drafting in local use
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/CliMA/ClimaTimeSteppers.jl.git",
    target = "build",
    push_preview = true,
    devbranch = "main",
    forcepush = true,
)
