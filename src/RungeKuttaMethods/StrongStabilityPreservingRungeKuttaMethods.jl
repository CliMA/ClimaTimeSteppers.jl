export StrongStabilityPreservingRungeKutta
export SSPRK22Heuns, SSPRK22Ralstons, SSPRK33ShuOsher, SSPRK34SpiteriRuuth

"""
    StrongStabilityPreservingRungeKutta(f, RKA, RKB, RKC, Q; dt, t0 = 0)

This is a time stepping object for explicitly time stepping the differential
equation given by the right-hand-side function `f` with the state `Q`, i.e.,

```math
  \\dot{Q} = f(Q, t)
```

with the required time step size `dt` and optional initial time `t0`.  This
time stepping object is intended to be passed to the `solve!` command.

The constructor builds a strong-stability-preserving Runge--Kutta scheme
based on the provided `RKA`, `RKB` and `RKC` coefficient arrays.

The available concrete implementations are:

  - [`SSPRK22Heuns`](@ref)
  - [`SSPRK22Ralstons`](@ref)
  - [`SSPRK33ShuOsher`](@ref)
  - [`SSPRK34SpiteriRuuth`](@ref)
"""
abstract type StrongStabilityPreservingRungeKutta <: DistributedODEAlgorithm end

struct StrongStabilityPreservingRungeKuttaTableau{Nstages, RT}
    "Storage RK coefficient vector A1 (rhs scaling of u)"
    A1::NTuple{Nstages, RT}
    "Storage RK coefficient vector A2 (rhs scaling of U)"
    A2::NTuple{Nstages, RT}
    "Storage RK coefficient vector B (rhs add in scaling)"
    B::NTuple{Nstages, RT}
    "Storage RK coefficient vector C (time scaling)"
    C::NTuple{Nstages, RT}
end

struct StrongStabilityPreservingRungeKuttaCache{Nstages, RT, A}    
    tableau::StrongStabilityPreservingRungeKuttaTableau{Nstages, RT}
    "Storage for RHS during the `StrongStabilityPreservingRungeKutta` update"
    fU::A
    "Storage for the stage state during the `StrongStabilityPreservingRungeKutta` update"
    U::A
end

function cache(
    prob::DiffEqBase.AbstractODEProblem{uType, tType, true}, 
    alg::StrongStabilityPreservingRungeKutta; kwargs...) where {uType,tType}

    tab = tableau(alg, eltype(prob.u0))
    # can't use Vector{T}(undef) as need to ensure no NaNs
    fU = zero(prob.u0)
    U = zero(prob.u0)
    return StrongStabilityPreservingRungeKuttaCache(tab, fU, U)
end


function step_u!(int, cache::StrongStabilityPreservingRungeKuttaCache{Nstages, RT, A}) where {Nstages, RT, A}
    tab = cache.tableau

    f! = int.prob.f
    u = int.u
    p = int.prob.p
    t = int.t
    dt = int.dt

    
    for s in 1:Nstages
        if s == 1
            f!(cache.fU, u, p, t + tab.C[s]*dt)
        else
            f!(cache.fU, cache.U, p, t + tab.C[s]*dt)
        end
        if s < Nstages
            cache.U .= tab.A1[s] .* u  .+  tab.A2[s] .* cache.U .+ (dt * tab.B[s]) .* cache.fU
        else
            u .= tab.A1[s] .* u  .+  tab.A2[s] .* cache.U .+ (dt * tab.B[s]) .* cache.fU
        end
    end
end


"""
    SSPRK22Heuns()

The second-order, 2-stage, strong-stability-preserving, Runge--Kutta scheme
of Shu and Osher (1988) (also known as Heun's method.)
Exact choice of coefficients from wikipedia page for Heun's method :)

### References
    @article{shu1988efficient,
      title={Efficient implementation of essentially non-oscillatory shock-capturing schemes},
      author={Shu, Chi-Wang and Osher, Stanley},
      journal={Journal of computational physics},
      volume={77},
      number={2},
      pages={439--471},
      year={1988},
      publisher={Elsevier}
    }
    @article {Heun1900,
       title = {Neue Methoden zur approximativen Integration der
       Differentialgleichungen einer unabh\"{a}ngigen Ver\"{a}nderlichen}
       author = {Heun, Karl},
       journal = {Z. Math. Phys},
       volume = {45},
       pages = {23--38},
       year = {1900}
    }
"""
struct SSPRK22Heuns <: StrongStabilityPreservingRungeKutta end

function tableau(::SSPRK22Heuns, RT)
    RKA1 = (RT(1), RT(1 // 2))
    RKA2 = (RT(0), RT(1 // 2))
    RKB  = (RT(1), RT(1 // 2))
    RKC  = (RT(0), RT(1))
    StrongStabilityPreservingRungeKuttaTableau(RKA1, RKA2, RKB, RKC)
end

"""
    SSPRK22Ralstons()

The second-order, 2-stage, strong-stability-preserving, Runge--Kutta scheme
of Shu and Osher (1988) (also known as Ralstons's method.)
Exact choice of coefficients from wikipedia page for Heun's method :)

### References
    @article{shu1988efficient,
      title={Efficient implementation of essentially non-oscillatory shock-capturing schemes},
      author={Shu, Chi-Wang and Osher, Stanley},
      journal={Journal of computational physics},
      volume={77},
      number={2},
      pages={439--471},
      year={1988},
      publisher={Elsevier}
    }
    @article{ralston1962runge,
      title={Runge-Kutta methods with minimum error bounds},
      author={Ralston, Anthony},
      journal={Mathematics of computation},
      volume={16},
      number={80},
      pages={431--437},
      year={1962},
      doi={10.1090/S0025-5718-1962-0150954-0}
    }
"""
struct SSPRK22Ralstons <: StrongStabilityPreservingRungeKutta end

function tableau(::SSPRK22Ralstons, RT)
    RKA1 = (RT(1), RT(5 // 8)) 
    RKA2 = (RT(0), RT(3 // 8))
    RKB  = (RT(2 // 3), RT(3 // 4))
    RKC  = (RT(0), RT(2 // 3))
    StrongStabilityPreservingRungeKuttaTableau(RKA1, RKA2, RKB, RKC)
end

"""
    SSPRK33ShuOsher()

The third-order, 3-stage, strong-stability-preserving, Runge--Kutta scheme
of Shu and Osher (1988)

### References
    @article{shu1988efficient,
      title={Efficient implementation of essentially non-oscillatory shock-capturing schemes},
      author={Shu, Chi-Wang and Osher, Stanley},
      journal={Journal of computational physics},
      volume={77},
      number={2},
      pages={439--471},
      year={1988},
      publisher={Elsevier}
    }
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
of Spiteri and Ruuth (1988)

### References
    @article{spiteri2002new,
      title={A new class of optimal high-order strong-stability-preserving time discretization methods},
      author={Spiteri, Raymond J and Ruuth, Steven J},
      journal={SIAM Journal on Numerical Analysis},
      volume={40},
      number={2},
      pages={469--491},
      year={2002},
      publisher={SIAM}
    }
"""
struct SSPRK34SpiteriRuuth <: StrongStabilityPreservingRungeKutta end

function tableau(::SSPRK34SpiteriRuuth, RT)
    RKA1 = (RT(1), RT(0), RT(2 // 3), RT(0)) 
    RKA2 = (RT(0), RT(1), RT(1 // 3), RT(1))
    RKB  = (RT(1 // 2), RT(1 // 2), RT(1 // 6), RT(1 // 2))
    RKC  = (RT(0), RT(1 // 2), RT(1), RT(1 // 2))
    StrongStabilityPreservingRungeKuttaTableau(RKA1, RKA2, RKB, RKC)
end
