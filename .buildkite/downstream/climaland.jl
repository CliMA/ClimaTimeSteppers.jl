#=
Downstream integration test: ClimaLand.jl

Runs the integrated full_land test to verify that CTS changes don't
break the ClimaLand integration path. This test exercises SciMLBase.solve
with CTS.IMEXAlgorithm(ARS111(), NewtonsMethod(...)).
=#
@info "ClimaLand: running full_land test..."
import ClimaLand
include(joinpath(pkgdir(ClimaLand), "test", "integrated", "full_land.jl"))
@info "ClimaLand: completed successfully"
