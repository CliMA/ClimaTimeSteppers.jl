# This file implements the Ros-SSP scheme as described in (Hai and Yagi, 2012),
# page 409-410. Note that the paper has some typos.
#
# We modify the scheme to apply horizontal limiters.

# Here, we use the coefficients on page 411 for RosERK(3,2), alphas and betas
# are from previous pages (with 1/!2 = 1/12)

import ClimaTimeSteppers
import LinearAlgebra: ldiv!, lu

abstract type RosSSPAlgorithmName <: ClimaTimeSteppers.AbstractAlgorithmName end

"""
    RosenbrockTableau

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
    RosenbrockAlgorithm(tableau)

Constructs a Rosenbrock algorithm for solving ODEs.
"""
struct RosSSPAlgorithm{T <: RosSSPTableau} <: ClimaTimeSteppers.DistributedODEAlgorithm
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

function ClimaTimeSteppers.init_cache(prob, alg::RosSSPAlgorithm; kwargs...)
    num_stages = length(alg.tableau.bI)
    KI = ntuple(_ -> zero(prob.u0), num_stages + 1)
    Y = ntuple(_ -> zero(prob.u0), num_stages + 1)
    fU_exp = ntuple(_ -> zero(prob.u0), num_stages + 1)
    fU_imp = ntuple(_ -> zero(prob.u0), num_stages + 1)
    fU_lim = ntuple(_ -> zero(prob.u0), num_stages + 1)
    U_imp = zero(prob.u0)
    KI_rhs = zero(prob.u0)
    if !isnothing(prob.f.T_imp!)
        W = prob.f.T_imp!.jac_prototype
        J = similar(W.matrix)
    else
        W = nothing
        J = nothing
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

function ClimaTimeSteppers.step_u!(int, cache::RosSSPCache)
    (; u, p, t, dt) = int
    f = int.sol.prob.f
    T_imp! = !isnothing(f.T_imp!) ? f.T_imp! : (args...) -> nothing
    T_exp! = !isnothing(f.T_exp!) ? f.T_exp! : (args...) -> nothing

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

    if !isnothing(f.T_imp!)
        # FIXME: Assumes βI has same values on diagonal
        @assert length(unique(βI[i, i] for i in 1:(num_stages - 1) )) == 1

        dtγ = dt * βI[1, 1]
        Wfact! = int.sol.prob.f.T_imp!.Wfact
        Wfact!(W, u, p, dtγ, t)
        J .= (W.matrix .+ one(W.matrix)) ./ dtγ
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
                U_imp .= u_n .+ sum(αI[i, j] * KI[j] for j in 1:(i-1); init = zero(u_n)) .+ dt .* sum(αE[i, j] * fU_exp[j] for j in 1:(i-1); init = zero(u_n))
                T_imp!(fU_imp[i], U_imp, p, t + ci * dt)
                KI_rhs .= .- dt .* fU_imp[i] .- dt * J .* (sum(βI[i, j] * KI[j] for j in 1:(i-1); init = zero(u_n)) + dt * sum(βE[i, j] * fU_exp[j] for j in 1:(i-1); init = zero(u_n)))
                dss!(KI_rhs, p, t + ci * dt)
                if W isa Matrix
                    ldiv!(KI[i], lu(W), KI_rhs)
                else
                    ldiv!(KI[i], W, KI_rhs)
                end
            end
        else
            KI[i] .= zero(u_n)
        end

        if i == 1
            Y[1] .= u_n + ζI[1, 1] * KI[1]
        else
            Y[i] .= sum(η[i, j] * Y[j] + dt * ΘE[i, j] * fU_exp[j] for j in 1:(i-1); init = zero(u_n)) + sum(ΘI[i, j] * KI[j] for j in 1:i; init = zero(u_n))
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
        dss!(fU_exp[i], p, t + ci * dt)
        post_implicit!(Y[i], p, t + ci * dt)
    end
    u .= Y[num_stages + 1]
    return nothing
end
