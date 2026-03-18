#=
julia --project=test
using Revise; include("test/solvers/esdirk.jl")
=#

using ClimaTimeSteppers, LinearAlgebra, Test

if !@isdefined(IntegratorTestCase)
    include(joinpath(@__DIR__, "..", "utils", "convergence_utils.jl"))
end

import SciMLBase
import ClimaTimeSteppers as CTS

function esdirk_linear_stiff_test_cts(::Type{FT}) where {FT}
    λ = FT(-1000)
    Y₀ = FT[1]
    ClimaIntegratorTestCase(;
        test_name = "esdirk_linear_stiff",
        linear_implicit = true,
        t_end = FT(0.01),
        Y₀,
        analytic_sol = (t) -> [exp(λ * t)],
        tendency! = (Yₜ, Y, _, t) -> Yₜ .= λ .* Y,
        Wfact! = (W, Y, _, dtγ, t) -> W .= dtγ .* λ .- 1,
        default_num_steps = 250,
    )
end

@testset "ESDIRK convergence (implicit-only)" begin
    test_convergence!(CTS.ESDIRK43(), esdirk_linear_stiff_test_cts(Float64), 250)
end

