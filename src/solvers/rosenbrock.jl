abstract type RosenbrockAlgorithm <: DistributedODEAlgorithm end

struct RosenbrockTableau{N, RT, N²}
    A::SMatrix{N, N, RT, N²}
    C::SMatrix{N, N, RT, N²}
    Γ::SMatrix{N, N, RT, N²}
    m::SVector{N, RT}
end

struct RosenbrockCache{Nstages, RT, N², A}
    tableau::RosenbrockTableau{Nstages, RT, N²}
    fU::A
    k::NTuple{Nstages, A}
end

function cache(
    prob::DiffEqBase.AbstractODEProblem{uType, tType, true},
    alg::RosenbrockAlgorithm; kwargs...) where {uType,tType}

    tab = tableau(alg, eltype(prob.u0))
    Nstages = length(tab.m)
    fU = zero(prob.u0)
    k = ntuple(n -> similar(prob.u0), Nstages)
    return RosenbrockCache(tab, fU, k)
end


function step_u!(int, cache::StrongStabilityPreservingRungeKuttaCache{Nstages, RT, A}) where {Nstages, RT, A}
    tab = cache.tableau

    f! = int.prob.f
    Wfact! = int.prob.f.Wfact

    u = int.u
    p = int.prob.p
    t = int.t
    dt = int.dt

    # 1) compute jacobian factorization
    γ = dt * tab.Γ[1,1]
    Wfact!(W, u, p, γ, t)

    for i in 1:Nstages
        U .= u
        for j = 1:i-1
            U .+= tab.A[i,j] .* k[j]
        end
        # TODO: there should be a time modification here (t + c * dt)
        # if f does depend on time, would need to add tgrad term as well
        f!(fU, U, p, t)
        for j = 1:i-1
            fU .+= (tab.C[i,j] / dt) .* k[j]
        end
        linsolve!(k[j], W, fU)
    end
    for i = 1:Nstages
        u .+= tab.m[i] .* k[i]
    end
end

struct SSPKnoth <: RosenbrockAlgorithm end

function tableau(::SSPKnoth, RT)
  # ROS.transformed=true;
    N = 3
    α = @SMatrix RT[
      0 0 0;
      1 0 0;
      1/4 1/4 0]
    # ROS.d=ROS.alpha*ones(ROS.nStage,1);
    b = @SVector RT[1/6 1/6 2/3]
    Γ = @SVector RT[
        1 0 0;
        0 1 0;
        -3/4 -3/4 1]
    A = α / Γ
    C = -inv(Γ)
    m = b / Γ
    return RosenbrockTableau{N, RT, N²}(A, C, Γ, m)
#   ROS.SSP.alpha=[1 0 0
#                  3/4 1/4 0
#                  1/3 0 2/3];

end
