export SSPKnoth

abstract type RosenbrockAlgorithm <: DistributedODEAlgorithm end

struct RosenbrockTableau{N, RT, N²}
    A::SMatrix{N, N, RT, N²}
    C::SMatrix{N, N, RT, N²}
    Γ::SMatrix{N, N, RT, N²}
    m::SMatrix{N, 1, RT, N}
end

struct RosenbrockCache{Nstages, RT, N², A}
    tableau::RosenbrockTableau{Nstages, RT, N²}
    U::A
    fU::A
    k::NTuple{Nstages, A}
    W::Any
    linsolve!::Any
end

function cache(prob::DiffEqBase.AbstractODEProblem, alg::RosenbrockAlgorithm; kwargs...)

    tab = tableau(alg, eltype(prob.u0))
    Nstages = length(tab.m)
    U = zero(prob.u0)
    fU = zero(prob.u0)
    k = ntuple(n -> similar(prob.u0), Nstages)
    W = prob.f.jac_prototype
    linsolve! = alg.linsolve(Val{:init}, W, prob.u0; kwargs...)

    return RosenbrockCache(tab, U, fU, k, W, linsolve!)
end


function step_u!(int, cache::RosenbrockCache{Nstages, RT}) where {Nstages, RT}
    tab = cache.tableau

    f! = int.sol.prob.f
    Wfact_t! = int.sol.prob.f.Wfact_t

    u = int.u
    p = int.p
    t = int.t
    dt = int.dt
    W = cache.W
    U = cache.U
    fU = cache.fU
    k = cache.k
    linsolve! = cache.linsolve!

    # 1) compute jacobian factorization
    γ = dt * tab.Γ[1, 1]
    Wfact_t!(W, u, p, γ, t)
    for i in 1:Nstages
        U .= u
        for j in 1:(i - 1)
            U .+= tab.A[i, j] .* k[j]
        end
        # TODO: there should be a time modification here (t + c * dt)
        # if f does depend on time, would need to add tgrad term as well
        f!(fU, U, p, t)
        for j in 1:(i - 1)
            fU .+= (tab.C[i, j] / dt) .* k[j]
        end
        linsolve!(k[i], W, fU)
    end
    for i in 1:Nstages
        u .+= tab.m[i] .* k[i]
    end
end

struct SSPKnoth{L} <: RosenbrockAlgorithm
    linsolve::L
end
SSPKnoth(; linsolve) = SSPKnoth(linsolve)


function tableau(::SSPKnoth, RT)
    # ROS.transformed=true;
    N = 3
    N² = N * N
    α = @SMatrix RT[
        0 0 0
        1 0 0
        1/4 1/4 0
    ]
    # ROS.d=ROS.alpha*ones(ROS.nStage,1);
    b = @SMatrix RT[1 / 6 1 / 6 2 / 3]
    Γ = @SMatrix RT[
        1 0 0
        0 1 0
        -3/4 -3/4 1
    ]
    A = α / Γ
    C = -inv(Γ)
    m = b / Γ
    return RosenbrockTableau{N, RT, N²}(A, C, Γ, m)
    #   ROS.SSP.alpha=[1 0 0
    #                  3/4 1/4 0
    #                  1/3 0 2/3];

end
