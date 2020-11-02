push!(LOAD_PATH,"..")

using Documenter, DocumenterCitations
using TimeMachine

bib = CitationBibliography(joinpath(@__DIR__, "refs.bib"))


makedocs(
    bib,
    sitename = "TimeMachine",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    modules = [TimeMachine],
    pages = [
        "index.md",
        "algorithms.md",
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
    repo = "github.com/CliMA/TimeMachine.jl.git",
    push_preview = true,
    forcepush = true,
)
