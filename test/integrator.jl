using ClimaTimeSteppers, Test

include("problems.jl")

@testset "saving default" begin
  sol = solve(deepcopy(const_prob), SSPRK33ShuOsher(); dt=1/32)
  ts = [const_prob.tspan[end]]
  @test sol.t == ts
  @test sol.u ≈ [const_sol(const_prob.u0, const_prob.p, t) for t in sol.t]
end
@testset "saveat isa Number" begin
  sol = solve(deepcopy(const_prob), SSPRK33ShuOsher(); dt=1/32, saveat=0.1)
  ts = 0:0.1:1
  @test all(abs.(sol.t .- ts) .< 1/32)
  @test all(isinteger, sol.t .* 32)
  @test sol.u ≈ [const_sol(const_prob.u0, const_prob.p, t) for t in sol.t]
end
@testset "saveat isa Range" begin
  ts = 0:0.1:1
  sol = solve(deepcopy(const_prob), SSPRK33ShuOsher(); dt=1/32, saveat=0:0.1:1)
  @test all(abs.(sol.t .- ts) .< 1/32)
  @test all(isinteger, sol.t .* 32)
  @test sol.u ≈ [const_sol(const_prob.u0, const_prob.p, t) for t in sol.t]
end
@testset "save_everystep" begin
  sol = solve(deepcopy(const_prob), SSPRK33ShuOsher(); dt=1/32, save_everystep=true)
  ts = 0:1/32:1
  @test sol.t == ts
  @test sol.u ≈ [const_sol(const_prob.u0, const_prob.p, t) for t in sol.t]
end
@testset "saveat too many" begin
  sol = solve(deepcopy(const_prob), SSPRK33ShuOsher(); dt=1/32, saveat=0:0.01:1)
  ts = 0:1/32:1
  @test sol.t == ts
  @test sol.u ≈ [const_sol(const_prob.u0, const_prob.p, t) for t in sol.t]
end