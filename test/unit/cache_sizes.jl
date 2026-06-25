using ClimaTimeSteppers, LinearAlgebra, Test
import ClimaTimeSteppers as CTS

@testset "Cache size verification" begin
    u0 = [1.0, 2.0]
    z0 = [0.0, 0.0] # init_cache allocates zero(u0)
    tspan = (0.0, 1.0)
    dt = 0.1

    # 1. IMEX ARK Unconstrained
    prob_imex = CTS.ODEProblem(
        ClimaODEFunction(;
            T_exp! = (du, u, p, t) -> (du .= -u),
            T_imp! = CTS.ODEFunction(
                (du, u, p, t) -> (du .= -u);
                jac_prototype = zeros(2, 2),
                Wfact = (W, u, p, γ, t) -> (W .= -I),
            ),
        ),
        u0, tspan, nothing,
    )
    int_ark = CTS.init(prob_imex, IMEXAlgorithm(ARS343(), NewtonsMethod(; max_iters=1)); dt)
    @test length(int_ark.cache.T_exp.data) == 4
    @test length(int_ark.cache.T_imp.data) == 3
    @test length(int_ark.cache.T_lim.data) == 4
    @test int_ark.cache.U == z0
    @test int_ark.cache.temp == z0
    @test int_ark.cache.newtons_method_cache.Δx == z0
    @test int_ark.cache.newtons_method_cache.f == z0

    # 2. SSP IMEX
    int_ssp = CTS.init(prob_imex, IMEXAlgorithm(SSP333(), NewtonsMethod(; max_iters=1), SSP()); dt)
    @test length(int_ssp.cache.T_imp.data) == 3
    @test int_ssp.cache.U == z0
    @test int_ssp.cache.U_exp == z0
    @test int_ssp.cache.U_lim == z0
    @test int_ssp.cache.T_exp == z0
    @test int_ssp.cache.T_lim == z0
    @test int_ssp.cache.temp == z0

    # 3. Rosenbrock
    int_ros = CTS.init(prob_imex, CTS.RosenbrockAlgorithm(CTS.tableau(SSPKnoth())); dt)
    @test length(int_ros.cache.k) == 3
    @test int_ros.cache.U == z0
    @test int_ros.cache.fU == z0
    @test int_ros.cache.fU_imp == z0
    @test int_ros.cache.fU_exp == z0
    @test int_ros.cache.fU_lim == z0
    @test int_ros.cache.∂Y∂t == z0

    # 4. Low-Storage RK (2N)
    prob_lsrk = CTS.ODEProblem(
        CTS.IncrementingODEFunction{true}((du, u, p, t, α=true, β=false) -> (du .= α .* u .+ β .* du)),
        u0, tspan, nothing,
    )
    int_lsrk = CTS.init(prob_lsrk, LSRK54CarpenterKennedy(); dt)
    @test int_lsrk.cache.du == z0

    # 5. Multirate
    prob_mr = CTS.SplitODEProblem(
        CTS.IncrementingODEFunction{true}((du, u, p, t, α=true, β=false) -> (du .= α .* u .+ β .* du)),
        (du, u, p, t) -> (du .= -u),
        u0, tspan, nothing,
    )
    int_mr = CTS.init(prob_mr, Multirate(LSRK54CarpenterKennedy(), MIS3C()); dt=0.1, fast_dt=0.01)
    @test length(int_mr.cache.outercache.ΔU) == 3
    @test length(int_mr.cache.outercache.F) == 3
    @test int_mr.cache.innerinteg.cache.du == z0
end
