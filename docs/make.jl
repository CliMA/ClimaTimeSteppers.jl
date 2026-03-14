using Documenter, DocumenterCitations
using InteractiveUtils: subtypes
using ClimaTimeSteppers
using Literate

tutorial_basedir = "tutorials"
tutorial_basedir_from_here = joinpath(@__DIR__, "src", tutorial_basedir)

jl_files_in_basedir = filter(endswith(".jl"), readdir(tutorial_basedir_from_here))

println("Building literate tutorials...")
generated_tutorials = String[]
for filename in jl_files_in_basedir
    Literate.markdown(
        joinpath(tutorial_basedir_from_here, filename),
        tutorial_basedir_from_here;
        execute = true,
        flavor = Literate.CommonMarkFlavor(),
    )
    push!(
        generated_tutorials,
        joinpath(tutorial_basedir, replace(filename, ".jl" => ".md")),
    )
end

# https://github.com/jheinen/GR.jl/issues/278#issuecomment-587090846
ENV["GKSwstype"] = "nul"

# ── Generate convergence report plots ──────────────────────────────────────────
# Run the convergence report generator for every algorithm, then summarize the
# results into PNG plots placed in docs/src/assets/.
println("Generating convergence report plots...")
let report_dir = joinpath(@__DIR__, "src", "dev")
    include(joinpath(report_dir, "compute_convergence.jl"))
    include(joinpath(pkgdir(ClimaTimeSteppers), "test", "problems.jl"))
    # Collect all concrete algorithm names
    function all_subtypes(T)
        result = Type[]
        for s in subtypes(T)
            isabstracttype(s) ? append!(result, all_subtypes(s)) : push!(result, s)
        end
        result
    end
    for alg_type in all_subtypes(ClimaTimeSteppers.AbstractAlgorithmName)
        alg_str = string(nameof(alg_type))
        @info "Testing convergence of $alg_str"
        empty!(ARGS)
        push!(ARGS, "--alg", alg_str)
        try
            include(joinpath(report_dir, "report_gen_alg.jl"))
        catch e
            @warn "Convergence test failed for $alg_str, skipping" exception = e
        end
    end
    include(joinpath(report_dir, "summarize_convergence.jl"))
    # Copy generated PNGs into docs/src/assets/
    assets_dir = joinpath(@__DIR__, "src", "assets")
    mkpath(assets_dir)
    for f in filter(endswith(".png"), readdir("output"))
        cp(joinpath("output", f), joinpath(assets_dir, f); force = true)
    end
end

bib = CitationBibliography(joinpath(@__DIR__, "refs.bib"))

#! format: off
pages = [
    "index.md",
    "Algorithm Formulations" => [
        "ODE Solvers" => "algorithm_formulations/ode_solvers.md",
        "Newtons Method" => "algorithm_formulations/newtons_method.md",
        "Rosenbrock Method" => "algorithm_formulations/rosenbrock.md",
        "Low-Storage RK" => "algorithm_formulations/lsrk.md",
        "Multirate RK" => "algorithm_formulations/mrrk.md",
    ],
    "Algorithm Properties" => [
        "Stability" => "algorithm_properties/stability.md",
        "Convergence" => "algorithm_properties/convergence.md",
    ],
    "Tutorials" => generated_tutorials,
    "API docs" => [
        "ODE Solvers" => "api/ode_solvers.md",
        "Newtons Method" => "api/newtons_method.md",
        "Callbacks" => "api/callbacks.md",
    ],
    # "Algorithm comparisons" => "algo_comparisons.md", # TODO: fill out
    "Developer docs" => [
        "Types" => "dev/types.md",
        "Developer Guide" => "dev/report_gen.md",
    ],
    "references.md",
]
#! format: on

mathengine = MathJax(
    Dict(:TeX => Dict(:equationNumbers => Dict(:autoNumber => "AMS"), :Macros => Dict())),
)

format = Documenter.HTML(
    prettyurls = get(ENV, "CI", nothing) == "true",
    mathengine = mathengine,
    collapselevel = 1,
)

makedocs(;
    plugins = [bib],
    sitename = "ClimaTimeSteppers",
    format = format,
    modules = [ClimaTimeSteppers],
    checkdocs = :exports,
    clean = true,
    pages = pages,
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
