# This file implements the Ros-SSP scheme as described in (Hai and Yagi, 2012),
# page 409-410. Note that the paper has some typos.
#
# We modify the scheme to apply horizontal limiters.

# Here, we use the coefficients on page 411 for RosERK(3,2), alphas and betas
# are from previous pages (with 1/!2 = 1/12)

import LinearAlgebra: ldiv!, lu, UniformScaling

import ClimaCore
import ClimaCore.MatrixFields: @name

export DaiYagi

abstract type RosSSPAlgorithmName <: AbstractAlgorithmName end

"""
    RosSSPTableau

Contains everything that defines a RosSSP method.

Refer to the documentation for the precise meaning of the symbols below.
"""
struct RosSSPTableau
    η
    ΘI
    ΘE
    αI
    βI
    αE
    βE
    γE
    γI
    bI
    bE
    ζI
end


"""
    RosSSPAlgorithm(tableau)

Constructs a RosSSP algorithm for solving ODEs.
"""
struct RosSSPAlgorithm{T <: RosSSPTableau} <: DistributedODEAlgorithm
    tableau::T
end

struct DaiYagi <: RosSSPAlgorithmName end

function tableau(::DaiYagi)
    # NOTE: These coefficients are all related.
    # In the next version, I will compute one from another.

    η  = [ 0    0       0     0;
           1    0       0     0;
          1//4  3//4    0     0;
          1//4  1//4   1//2   0 ]

    ΘI = [  1//3     0       0    0;
           -2//3    1//3     0    0;
            2//3   -1//12   1//3  0;
            1//4     0     1//6   0 ]

    ΘE = [   0        0       0     0;
             1        0       0     0;
           -1//4     1//2     0     0;
             0      -1//12   1//3   0 ]

    αI = [   0        0       0;
           -1//6      0       0;
            1//4     1//12    0 ]

    βI = [  1//3      0       0;
           -1//6     1//3     0;
            1//4     1//12   1//3 ]

    αE = [   0        0       0;
            1//2      0       0;
            1//4     1//4     0 ]

    βE = [   0        0       0;
            1//2      0       0;
            1//4     1//4     0 ]

    γE = [   0        0       0;
             1        0       0;
            1//2     1//2     0 ]

    γI = [  1//3      0       0;
           -1//3     1//3     0;
            1//2     1//6    1//3 ]

    bI = [ 1//2  1//6  1//3 ]
    bE = [ 1//2  1//6  1//3 ]

    ζI = vcat(γI, bI)

    return RosSSPTableau(
    η ,
    ΘI,
    ΘE,
    αI,
    βI,
    αE,
    βE,
    γE,
    γI,
    bI,
    bE,
    ζI
    )
end

struct RosSSPCache
    num_stages
    KI
    Y
    J
    fU_exp
    fU_imp
    fU_lim
    U_imp
    KI_rhs
    W
end

function init_cache(prob, alg::RosSSPAlgorithm; kwargs...)
    num_stages = length(alg.tableau.bI)
    KI = ntuple(_ -> NaN32 * zero(prob.u0), num_stages + 1)
    Y = ntuple(_ -> NaN32 * zero(prob.u0), num_stages + 1)
    fU_exp = ntuple(_ -> NaN32 * zero(prob.u0), num_stages + 1)
    fU_imp = ntuple(_ -> NaN32 * zero(prob.u0), num_stages + 1)
    fU_lim = ntuple(_ -> NaN32 * zero(prob.u0), num_stages + 1)
    U_imp = zero(prob.u0)
    KI_rhs = zero(prob.u0)
    if !isnothing(prob.f.T_imp!)
        W = prob.f.T_imp!.jac_prototype
        J = similar(W.matrix)
    else
        W = nothing
        J = nothing
        dtγJ = nothing
    end
    return RosSSPCache(
        num_stages,
        KI,
        Y,
        J,
        fU_exp,
        fU_imp,
        fU_lim,
        U_imp,
        KI_rhs,
        W
    )
end

function step_u!(int, cache::RosSSPCache)
    (; u, p, t, dt) = int
    f = int.sol.prob.f
    T_imp! = !isnothing(f.T_imp!) ? f.T_imp! : (args...) -> nothing
    T_exp! = !isnothing(f.T_exp!) ? f.T_exp! : (args...) -> nothing
    T_exp_T_lim! = !isnothing(f.T_exp_T_lim!) ? f.T_exp_T_lim! : (args...) -> nothing

    (; post_explicit!, post_implicit!, dss!) = int.sol.prob.f

    (; num_stages,
     KI,
     Y,
     J,
     fU_exp,
     fU_imp,
     fU_lim,
     U_imp,
     KI_rhs,
     W) = cache

    (; η ,
       ΘI,
       ΘE,
       αI,
       βI,
       αE,
       βE,
       γE,
       γI,
       bI,
       bE,
       ζI ) = int.alg.tableau

    u_n = copy(u)

    # FIXME: Assumes βI has same values on diagonal
    @assert length(unique(βI[i, i] for i in 1:(num_stages - 1) )) == 1
    dtγ = dt * βI[1, 1]

    if !isnothing(f.T_imp!)
        Wfact! = int.sol.prob.f.T_imp!.Wfact
        Wfact!(W, u, p, dtγ, t)

        center_field = u.c.ρ
        face_field = u.f.u₃
        FT = eltype(center_field)
        myJ = ClimaCore.MatrixFields.FieldMatrix((@name(c.ρ), @name(c.ρ)) => similar(center_field, ClimaCore.MatrixFields.DiagonalMatrixRow{FT})
                                                , (@name(c.ρ), @name(f.u₃)) => W.matrix[@name(c.ρ), @name(f.u₃)]
                                                , (@name(c.ρe_tot), @name(f.u₃)) => W.matrix[@name(c.ρe_tot), @name(f.u₃)]
                                                , (@name(f.u₃), @name(c.ρ)) => W.matrix[@name(f.u₃), @name(c.ρ)]
                                                , (@name(f.u₃), @name(c.ρe_tot)) => W.matrix[@name(f.u₃), @name(c.ρe_tot)]
                                                , (@name(f.u₃), @name(c.uₕ)) => W.matrix[@name(f.u₃), @name(c.uₕ)]
                                                , (@name(f.u₃), @name(f.u₃)) => W.matrix[@name(f.u₃), @name(f.u₃)]
                                                , (@name(c.ρe_tot), @name(c.ρe_tot)) =>  similar(center_field, ClimaCore.MatrixFields.DiagonalMatrixRow{FT})
                                                , (@name(c.uₕ), @name(c.uₕ)) =>  similar(center_field, ClimaCore.MatrixFields.DiagonalMatrixRow{FT}
                                                  ))

        myone = ClimaCore.MatrixFields.FieldMatrix((@name(c.ρ), @name(c.ρ)) => LinearAlgebra.I
                                                , (@name(f.u₃), @name(f.u₃)) =>
                                                    ones(ClimaCore.MatrixFields.DiagonalMatrixRow{eltype(eltype(W.matrix[@name(f.u₃), @name(f.u₃)]))}, axes(face_field))
                                                , (@name(c.ρe_tot), @name(c.ρe_tot)) => LinearAlgebra.I
                                                , (@name(c.uₕ), @name(c.uₕ)) => LinearAlgebra.I)

        # dtγ_matrix = ClimaCore.MatrixFields.FieldMatrix((ClimaCore.MatrixFields.@name(),
        #                                                  ClimaCore.MatrixFields.@name()) => 1/(dtγ) * LinearAlgebra.I
        #                                                             )

        mydtγ = ClimaCore.MatrixFields.FieldMatrix((@name(c.ρ), @name(c.ρ)) => LinearAlgebra.I / dtγ
                                                , (@name(f.u₃), @name(f.u₃)) =>
                                                    ClimaCore.MatrixFields.DiagonalMatrixRow.(ones(eltype(eltype(W.matrix[@name(f.u₃), @name(f.u₃)])), axes(face_field)) ./ dtγ)
                                                , (@name(c.ρe_tot), @name(c.ρe_tot)) => LinearAlgebra.I / dtγ
                                                , (@name(c.uₕ), @name(c.uₕ)) => LinearAlgebra.I / dtγ)

        myJ .= (W.matrix .+ myone) .* mydtγ
    end

    if !isnothing(f.T_exp!) && !isnothing(f.T_exp_T_lim!)
        error("Problem")
        # We don't accumulate T_exp in the function calls below
    end

    for i in 1:(num_stages + 1)
        # FIXME: Check both EXP-IMP tableau for ci
        if i == num_stages + 1
            ci = 1
        else
            ci = sum(αI[i, 1:(i - 1)])
        end

        if !isnothing(f.T_imp!)
            if i == num_stages + 1
                KI[i] .= zero(u_n)
            else
                # @show αI[3, 1] * KI[1] +  αI[3, 2] * KI[2]
                U_imp .= u_n .+ sum(αI[i, j] * KI[j] for j in 1:(i-1); init = zero(u_n)) .+ dt .* sum(αE[i, j] * fU_exp[j] for j in 1:(i-1); init = zero(u_n))
                T_imp!(fU_imp[i], U_imp, p, t + ci * dt)
                mysum = (sum(βI[i, j] * KI[j] for j in 1:(i-1); init = zero(u_n)) + dt * sum(βE[i, j] * fU_exp[j] for j in 1:(i-1); init = zero(u_n)))
                dt_matrix = ClimaCore.MatrixFields.FieldMatrix((@name(c.ρ), @name(c.ρ)) => LinearAlgebra.I * (-dt)
                                                               , (@name(f.u₃), @name(f.u₃)) =>
                                                                   ClimaCore.MatrixFields.DiagonalMatrixRow.(ones(eltype(eltype(W.matrix[@name(f.u₃), @name(f.u₃)])), axes(face_field)) .* (-dt))
                                                               , (@name(c.ρe_tot), @name(c.ρe_tot)) => LinearAlgebra.I * (-dt)
                                                               , (@name(c.uₕ), @name(c.uₕ)) => LinearAlgebra.I * (-dt))

                # dt_matrix = ClimaCore.MatrixFields.FieldMatrix((ClimaCore.MatrixFields.@name(),
                #                                                      ClimaCore.MatrixFields.@name()) => dt * LinearAlgebra.I
                #                                                     )
                # oγ_matrix = ClimaCore.MatrixFields.FieldMatrix((ClimaCore.MatrixFields.@name(),
                #                                                      ClimaCore.MatrixFields.@name()) => 1/dtγ * LinearAlgebra.I
                #                                                     )
                KI_rhs .= dt_matrix .* fU_imp[i] .- myJ .* mysum
                dss!(KI_rhs, p, t + ci * dt)
                if W isa Matrix
                    ldiv!(KI[i], lu(W), KI_rhs)
                else
                    ldiv!(KI[i], W, KI_rhs)
                end
                # @show myJ[@name(c.ρ), @name(c.ρ)]

                if all(isnan, parent(KI[i].c.ρ))
                    fill!(KI[i].c.ρ, 0)
                end
                if all(isnan, parent(KI[i].c.ρe_tot))
                    fill!(KI[i].c.ρe_tot, 0)
                end
                if all(isnan, parent(KI[i].f.u₃))
                    fill!(parent(KI[i].f.u₃), 0)
                end

                # Main.@infiltrate i == 1

                # @show i, extrema(u_n), extrema(U_imp), extrema(fU_imp[i]), extrema(mysum), extrema(KI_rhs), extrema(KI[i])
            end
            # @show W, U_imp[i], KI_rhs[i], KI[i], i
        else
            KI[i] .= zero(u_n)
        end

        if i == 1
            Y[1] .= u_n .+ ζI[1, 1] * KI[1]
            # @show extrema(fU_exp[1]), extrema(Y[1])
        else
            # @show sum(ΘI[i, j] * KI[j] for j in 1:i; init = zero(u_n))
            Y[i] .= sum(η[i, j] .* Y[j] + dt .* ΘE[i, j] .* fU_exp[j] for j in 1:(i-1); init = zero(u_n)) #.+ sum(ΘI[i, j] * KI[j] for j in 1:i; init = zero(u_n))
        end

        if !isnothing(f.T_exp!)
            T_exp!(fU_exp[i], Y[i], p, t + ci * dt)
        else
            fU_exp[i] .= zero(u_n)
        end
        if !isnothing(f.T_exp_T_lim!)
            T_exp_T_lim!(fU_exp[i], fU_lim[i], Y[i], p, t + ci * dt)
            fU_exp[i] .+= fU_lim[i]
        else
            fU_lim[i] .= zero(u_n)
        end
        # @show i, extrema(Y[i].c.uₕ.components), extrema(fU_exp[i].c.uₕ.components)
        # @show i, extrema(Y[i].f.u₃.components), extrema(fU_exp[i].f.u₃.components), extrema(fU_exp[i])
        # @show i, extrema(Y[i].c.ρ), extrema(fU_exp[i].c.ρ)
        dss!(fU_exp[i], p, t + ci * dt)
        post_implicit!(Y[i], p, t + ci * dt)
    end
    u .= Y[num_stages + 1]

    # Main.@infiltrate true

    dss!(u, p, t + dt)
    post_implicit!(u, p, t + dt)
    return nothing
end
