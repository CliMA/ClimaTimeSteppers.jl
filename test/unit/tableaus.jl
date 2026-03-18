#=
Tableau consistency checks: verify mathematical properties of all Butcher tableaus.
=#
using ClimaTimeSteppers, Test, LinearAlgebra
using StaticArrays
import ClimaTimeSteppers as CTS

include(joinpath(@__DIR__, "..", "utils", "all_subtypes.jl"))

# Helper to extract dense Float64 matrix from SparseCoeffs
dense(sc::CTS.SparseCoeffs) = Array(Float64.(sc.coeffs))

@testset "Tableau consistency" begin

    # ========================================================================
    # Explicit tableaus
    # ========================================================================
    @testset "Explicit tableaus" begin
        for name_type in all_subtypes(CTS.ERKAlgorithmName)
            name = name_type()
            tab = ExplicitTableau(name)
            a = dense(tab.a)
            b = dense(tab.b)
            c = dense(tab.c)
            s = size(a, 1)

            @testset "$(nameof(name_type))" begin
                # a must be strictly lower triangular
                @test all(iszero, UpperTriangular(a))

                # First stage evaluates at t_n
                @test c[1] ≈ 0 atol = 1e-14

                # Row-sum consistency: c[i] == sum(a[i,:])
                for i in 1:s
                    @test c[i] ≈ sum(a[i, :]) atol = 1e-14
                end

                # Weights sum to 1 (consistency condition for order >= 1)
                @test sum(b) ≈ 1.0 atol = 1e-14
            end
        end
    end

    # ========================================================================
    # IMEX tableaus
    # ========================================================================
    @testset "IMEX tableaus" begin
        for name_type in all_subtypes(CTS.IMEXARKAlgorithmName)
            name = name_type()
            tab = CTS.IMEXTableau(name)
            a_exp = dense(tab.a_exp)
            b_exp = dense(tab.b_exp)
            c_exp = dense(tab.c_exp)
            a_imp = dense(tab.a_imp)
            b_imp = dense(tab.b_imp)
            c_imp = dense(tab.c_imp)
            s = size(a_exp, 1)

            @testset "$(nameof(name_type))" begin
                # Explicit part must be strictly lower triangular
                @test all(iszero, UpperTriangular(a_exp))

                # Implicit part must be lower triangular (DIRK)
                @test all(iszero, UpperTriangular(a_imp) - Diagonal(a_imp))

                # Sizes must match
                @test size(a_exp) == size(a_imp)
                @test length(b_exp) == s
                @test length(b_imp) == s
                @test length(c_exp) == s
                @test length(c_imp) == s

                # Row-sum consistency for explicit part
                for i in 1:s
                    @test c_exp[i] ≈ sum(a_exp[i, :]) atol = 1e-13
                end

                # Row-sum consistency for implicit part
                for i in 1:s
                    @test c_imp[i] ≈ sum(a_imp[i, :]) atol = 1e-13
                end

                # Explicit weights sum to 1 when there is an explicit part.
                # Some purely implicit DIRK/ESDIRK algorithms are encoded with
                # an all-zero explicit tableau, in which case the sum condition
                # is not applicable.
                if !all(iszero, b_exp)
                    @test sum(b_exp) ≈ 1.0 atol = 1e-13
                end

                # Implicit weights sum to 1
                @test sum(b_imp) ≈ 1.0 atol = 1e-13
            end
        end
    end

    # ========================================================================
    # SDIRK consistency: implicit diagonal should be constant for SDIRK methods
    # ========================================================================
    @testset "SDIRK diagonal consistency" begin
        # These ARS methods are SDIRK (all nonzero diagonal entries are the same).
        # Update this list when new SDIRK methods are added.
        sdirk_names = [ARS122, ARS232, ARS222, ARS233, ARS343, ARS443]
        for name_type in sdirk_names
            name = name_type()
            tab = CTS.IMEXTableau(name)
            a_imp = dense(tab.a_imp)
            diag_vals = filter(!iszero, diag(a_imp))
            if !isempty(diag_vals)
                @testset "$(nameof(name_type)) SDIRK diagonal" begin
                    @test all(v -> v ≈ diag_vals[1], diag_vals)
                end
            end
        end
    end

    # ========================================================================
    # Weight non-negativity for SSP methods
    # ========================================================================
    @testset "SSP weight non-negativity" begin
        # Update this list when new SSP methods are added.
        ssp_names = [SSP222, SSP322, SSP332, SSP333, SSP433]
        for name_type in ssp_names
            name = name_type()
            tab = CTS.IMEXTableau(name)
            a_exp = dense(tab.a_exp)
            b_exp = dense(tab.b_exp)

            @testset "$(nameof(name_type))" begin
                @test all(>=(0), a_exp)
                @test all(>=(0), b_exp)
            end
        end
    end

    # ========================================================================
    # Parameter validation
    # ========================================================================
    @testset "SSP333 beta validation" begin
        # β must be strictly > 1/2 (assertion in IMEXTableau constructor)
        @test_throws AssertionError CTS.IMEXTableau(SSP333(β = 0.4))
        @test_throws AssertionError CTS.IMEXTableau(SSP333(β = 0.5))
    end

    # ========================================================================
    # Rosenbrock tableau
    # ========================================================================
    @testset "Rosenbrock tableaus" begin
        tab = ClimaTimeSteppers.tableau(SSPKnoth())
        @testset "SSPKnoth" begin
            # A = α * Γ⁻¹ should be strictly lower triangular
            @test all(iszero, UpperTriangular(tab.A))

            # α should be strictly lower triangular (explicit stages only)
            @test all(iszero, UpperTriangular(tab.α))

            # Γ should be lower triangular with nonzero diagonal
            @test all(iszero, UpperTriangular(tab.Γ) - Diagonal(tab.Γ))
            @test all(!iszero, diag(tab.Γ))

            # C = diag(1/γ_ii) - Γ⁻¹ should be strictly lower triangular
            @test all(x -> abs(x) < 1e-14, UpperTriangular(tab.C))
        end
    end
end
