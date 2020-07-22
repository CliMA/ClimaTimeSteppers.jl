push!(LOAD_PATH,"..")

using Documenter
using TimeMachine

makedocs(
    sitename = "TimeMachine",
    format = Documenter.HTML(),
    modules = [TimeMachine],
    pages = [
        "index.md",
        "ARK.md",
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
