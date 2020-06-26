"""
    ODESolvers

Ordinary differential equation solvers

JuliaDiffEq terminology:

* _Problem_: RHS function, initial u, time span, parameters
    * reuse ODEProblem from OrdinaryDiffEq
    * define new `IncrementODEProblem` for LSRK methods
  
  
* _Algorithm_: small objects (often singleton) which indicate what algorithm + options (e.g. linear solver type)
    * define new `DistributedODEAlgorithm`
  
* _Integrator_: contains everything necessary to solve. Used as:
  
      init(prob, alg, options...) => integrator
      step!(int) => runs single step
      solve!(int) => runs it to end
      solve(prob, alg, options...) => init + solve!
  
    *define new `DistributedODEIntegrator`
  
* _Solution_ (not implemented): contains the "solution" to the ODE.

"""
module ODESolvers

import DiffEqBase


include("problems.jl")

abstract type DistributedODEAlgorithm <: DiffEqBase.AbstractODEAlgorithm
end

include("integrators.jl")


# Include concrete implementations
include("RungeKuttaMethods/LowStorageRungeKuttaMethods.jl")
include("RungeKuttaMethods/StrongStabilityPreservingRungeKuttaMethods.jl")

end