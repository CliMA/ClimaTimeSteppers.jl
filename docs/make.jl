using Documenter, DocumenterCitations
using ClimaTimeSteppers

bib = CitationBibliography(joinpath(@__DIR__, "refs.bib"))


makedocs(
    bib,
    sitename = "ClimaTimeSteppers",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    modules = [ClimaTimeSteppers],
    pages = [
        "index.md",
        "algorithms.md",
        "newtons_method.md",
        "callbacks.md",
        "Background" => [
            "background/lsrk.md",
            "background/ssprk.md",
            "background/ark.md",
            "background/mrrk.md",
        ],
        "references.md",
    ]
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
