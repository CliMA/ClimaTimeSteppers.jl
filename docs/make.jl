push!(LOAD_PATH,"..")

using Documenter
using ODESolvers

makedocs(
    sitename = "ODESolvers",
    format = Documenter.HTML(),
    modules = [ODESolvers],
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
