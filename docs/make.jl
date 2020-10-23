push!(LOAD_PATH,"..")

using Documenter
using TimeMachine

makedocs(
    sitename = "TimeMachine",
    format = Documenter.HTML(),
    modules = [TimeMachine],
    pages = [
        "index.md",
        "algorithms.md",
        "callbacks.md",
        "Background" => [
            "background/LowStorageRungeKutta.md",
            "background/StrongStabilityPreservingRungeKutta.md",
            "background/AdditiveRungeKutta.md",
            "background/MultirateRungeKutta.md",
        ]
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/CliMA/TimeMachine.jl.git",
    push_preview = true,
)
