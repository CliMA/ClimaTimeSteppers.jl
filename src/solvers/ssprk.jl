export SSPRK22Heuns, SSPRK22Ralstons, SSPRK33ShuOsher, SSPRK34SpiteriRuuth

"""
    StrongStabilityPreservingRungeKutta <: DistributedODEAlgorithm

A class of Strong Stability Preserving Runge--Kutta methods.
These require two additional copies of the state vector.

The available implementations are:

  - [`SSPRK22Heuns`](@ref)
  - [`SSPRK22Ralstons`](@ref)
  - [`SSPRK33ShuOsher`](@ref)
  - [`SSPRK34SpiteriRuuth`](@ref)
"""
abstract type StrongStabilityPreservingRungeKutta <: DistributedODEAlgorithm end

struct StrongStabilityPreservingRungeKuttaTableau{T}
    "Storage RK coefficient vector A1 (rhs scaling of u)"
    A1::T
    "Storage RK coefficient vector A2 (rhs scaling of U)"
    A2::T
    "Storage RK coefficient vector B (rhs add in scaling)"
    B::T
    "Storage RK coefficient vector C (time scaling)"
    C::T
end
n_stages(::StrongStabilityPreservingRungeKuttaTableau{T}) where {T} = n_stages_ntuple(T)

struct StrongStabilityPreservingRungeKuttaCache{T <: StrongStabilityPreservingRungeKuttaTableau, A}
    tableau::T
    "Storage for RHS during the `StrongStabilityPreservingRungeKutta` update"
    fU::A
    "Storage for the stage state during the `StrongStabilityPreservingRungeKutta` update"
    U::A
end

n_stages(cache::StrongStabilityPreservingRungeKuttaCache) = n_stages(cache.tableau)

function init_cache(prob::DiffEqBase.AbstractODEProblem, alg::StrongStabilityPreservingRungeKutta; kwargs...)

    tab = tableau(alg, eltype(prob.u0))
    # can't use Vector{T}(undef) as need to ensure no NaNs
    fU = zero(prob.u0)
    U = zero(prob.u0)
    return StrongStabilityPreservingRungeKuttaCache(tab, fU, U)
end


function step_u!(int, cache::StrongStabilityPreservingRungeKuttaCache)
    Nstages = n_stages(cache)
    (; A1, A2, C, B) = cache.tableau
    (; u, p, t, dt, sol) = int
    f! = sol.prob.f

    if f! isa ForwardEulerODEFunction
        for s in 1:Nstages
            # U[s] = A1[s] * u + A2[s] * U[s-1] + B[s] * dt * f(U[s-1],p, t + C[s] * dt)
            #      = A1[s] * u + A2[s] * (U[s-1] + B[s]/A2[s] * dt * f(U[s-1],p, t + C[s] * dt))
            Un = s < Nstages ? cache.U : u

            # IncrementingODEFunction f!(ux, u, p, t, α, β) = ux .= α .* ux .+ β .* f(u,p,t)
            # We need                     un .= u .+ β .* f(u,p,t)
            if s == 1
                @assert A1[s] == 1 && A2[s] == 0
                f!(Un, u, p, t + C[s] * dt, B[s] * dt)
            else
                f!(cache.fU, cache.U, p, t + C[s] * dt, B[s] / A2[s] * dt)
                Un .= A1[s] .* u .+ A2[s] .* cache.fU
            end
            #@show Un
        end
    elseif f! isa DiffEqBase.IncrementingODEFunction
        for s in 1:Nstages
            # U[s] = A1[s] * u + A2[s] * U[s-1] + B[s] * dt * f(U[s-1],p, t + C[s] * dt)
            #      = (A1[s] * u + A2[s] * U[s-1]) + (B[s] * dt) * f(U[s-1],p, t + C[s] * dt))
            Un = s < Nstages ? cache.U : u

            # IncrementingODEFunction f!(ux, u, p, t, α, β) = ux .= α .* ux .+ β .* f(u,p,t)
            # We need                     un .= u .+ β .* f(u,p,t)
            if s == 1
                @assert A1[s] == 1 && A2[s] == 0
                Un .= u
                f!(Un, u, p, t + C[s] * dt, B[s] * dt, 1)
            else
                cache.fU .= A1[s] .* u .+ A2[s] .* cache.U
                f!(cache.fU, cache.U, p, t + C[s] * dt, B[s] * dt, 1)
                Un .= cache.fU
            end
        end
    else
        for s in 1:Nstages
            if s == 1
                f!(cache.fU, u, p, t + C[s] * dt)
            else
                f!(cache.fU, cache.U, p, t + C[s] * dt)
            end
            if s < Nstages
                cache.U .= A1[s] .* u .+ A2[s] .* cache.U .+ (dt * B[s]) .* cache.fU
            else
                u .= A1[s] .* u .+ A2[s] .* cache.U .+ (dt * B[s]) .* cache.fU
            end
        end
    end
end


"""
    SSPRK22Heuns()

The second-order, 2-stage, strong-stability-preserving, Runge--Kutta scheme of
[SO1988](@cite), also known as Heun's method ([Heun1900](@ref).

Exact choice of coefficients from wikipedia page for Heun's method :)
"""
struct SSPRK22Heuns <: StrongStabilityPreservingRungeKutta end

function tableau(::SSPRK22Heuns, RT)
    RKA1 = (RT(1), RT(1 // 2))
    RKA2 = (RT(0), RT(1 // 2))
    RKB = (RT(1), RT(1 // 2))
    RKC = (RT(0), RT(1))
    StrongStabilityPreservingRungeKuttaTableau(RKA1, RKA2, RKB, RKC)
end

"""
    SSPRK22Ralstons()

The second-order, 2-stage, strong-stability-preserving, Runge--Kutta scheme
of [SO1988](@cite), also known as Ralstons's method ([Rals1962](@cite).

Exact choice of coefficients from wikipedia page for Heun's method :)
"""
struct SSPRK22Ralstons <: StrongStabilityPreservingRungeKutta end

function tableau(::SSPRK22Ralstons, RT)
    RKA1 = (RT(1), RT(5 // 8))
    RKA2 = (RT(0), RT(3 // 8))
    RKB = (RT(2 // 3), RT(3 // 4))
    RKC = (RT(0), RT(2 // 3))
    StrongStabilityPreservingRungeKuttaTableau(RKA1, RKA2, RKB, RKC)
end

"""
    SSPRK33ShuOsher()

The third-order, 3-stage, strong-stability-preserving, Runge--Kutta scheme
of [SO1988](@cite).
"""
struct SSPRK33ShuOsher <: StrongStabilityPreservingRungeKutta end

function tableau(::SSPRK33ShuOsher, RT)
    RKA1 = (RT(1), RT(3 // 4), RT(1 // 3))
    RKA2 = (RT(0), RT(1 // 4), RT(2 // 3))
    RKB = (RT(1), RT(1 // 4), RT(2 // 3))
    RKC = (RT(0), RT(1), RT(1 // 2))
    StrongStabilityPreservingRungeKuttaTableau(RKA1, RKA2, RKB, RKC)
end

"""
    SSPRK34SpiteriRuuth()

The third-order, 4-stage, strong-stability-preserving, Runge--Kutta scheme
of [SR2002](@cite).
"""
struct SSPRK34SpiteriRuuth <: StrongStabilityPreservingRungeKutta end

function tableau(::SSPRK34SpiteriRuuth, RT)
    RKA1 = (RT(1), RT(0), RT(2 // 3), RT(0))
    RKA2 = (RT(0), RT(1), RT(1 // 3), RT(1))
    RKB = (RT(1 // 2), RT(1 // 2), RT(1 // 6), RT(1 // 2))
    RKC = (RT(0), RT(1 // 2), RT(1), RT(1 // 2))
    StrongStabilityPreservingRungeKuttaTableau(RKA1, RKA2, RKB, RKC)
end
