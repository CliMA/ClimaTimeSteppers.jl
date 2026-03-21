#=
Downstream integration test: ClimaAtmos.jl

Runs a single-column test (1 timestep) to verify that CTS changes don't
break the ClimaAtmos integration path.
=#
redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))

import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import Random
Random.seed!(1234)

# cd to the ClimaAtmos root so that any relative paths in configs resolve
cd(pkgdir(CA))

# Use a minimal config: single column, 1 second, 1 step
config_file = joinpath(
    pkgdir(CA),
    "config",
    "model_configs",
    "column_nogw_3d_test.yml",
)
config = CA.AtmosConfig(config_file; job_id = "cts_downstream_test")
simulation = CA.get_simulation(config)
(; integrator) = simulation

@info "ClimaAtmos: running 1 timestep..."
sol_res = CA.solve_atmos!(simulation)
@info "ClimaAtmos: completed successfully" t_final = integrator.t
