#=
Downstream integration test: ClimaCoupler.jl

Runs the slabplanet configuration to verify that CTS changes don't break
the coupled integration path. Uses the CI config (1 day, h_elem=4).
=#
ENV["CI"] = "true"

# The coupler repo path is passed via environment variable (may be relative to pwd)
coupler_dir = abspath(
    get(ENV, "CLIMACOUPLER_DIR", joinpath(@__DIR__, "..", "..", "..", "ClimaCoupler.jl")),
)

@info "ClimaCoupler: running slabplanet (1 day)..." coupler_dir
empty!(ARGS)
push!(ARGS,
    "--config_file",
    joinpath(coupler_dir, "config", "ci_configs", "slabplanet_default.yml"),
    "--job_id", "cts_downstream_test",
)
include(joinpath(coupler_dir, "experiments", "ClimaEarth", "run_amip.jl"))
@info "ClimaCoupler: completed successfully"
