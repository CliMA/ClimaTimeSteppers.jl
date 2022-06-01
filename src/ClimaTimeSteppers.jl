"""
    ClimaTimeSteppers

Ordinary differential equation solvers

JuliaDiffEq terminology:

* _Function_: the right-hand side function df/dt.
  * by default, a function gets wrapped in an `ODEFunction`
  * define new `IncrementingODEFunction` to support incrementing function calls.

* _Problem_: Function, initial u, time span, parameters and options

  du/dt = f(u,p,t) = fL(u,p,t)  + fR(u,p,t)

  fR(u,p,t) == f(u.p,t) - fL(u,p,t)
  fL(u,_,_) == A*u for some `A` (matrix free)

  SplitODEProlem(fL, fR)


  * `ODEProblem` from OrdinaryDiffEq.jl
    - use `jac` option to `ODEFunction` for linear + full IMEX (https://docs.sciml.ai/latest/features/performance_overloads/#ode_explicit_jac-1)
  * `SplitODEProblem` for linear + remainder IMEX
  * `MultirateODEProblem` for true multirate

* _Algorithm_: small objects (often singleton) which indicate what algorithm + options (e.g. linear solver type)
  * define new abstract `DistributedODEAlgorithm`, algorithms in this pacakge will be subtypes of this
  * define new `Multirate` for multirate solvers

* _Integrator_: contains everything necessary to solve. Used as:

  * define new `DistributedODEIntegrator` for solvers in this package

      init(prob, alg, options...) => integrator
      step!(int) => runs single step
      solve!(int) => runs it to end
      solve(prob, alg, options...) => init + solve!

* _Solution_ (not implemented): contains the "solution" to the ODE.


"""
module ClimaTimeSteppers


using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using LinearAlgebra
using StaticArrays
using CUDA
using MPI


array_device(::Union{Array, SArray, MArray}) = CPU()
array_device(::CuArray) = CUDADevice()
realview(x::Union{Array, SArray, MArray}) = x
realview(x::CuArray) = x


import DiffEqBase, SciMLBase, LinearAlgebra, DiffEqCallbacks

include("functions.jl")
include("operators.jl")

abstract type DistributedODEAlgorithm <: DiffEqBase.AbstractODEAlgorithm
end

SciMLBase.allowscomplex(alg::DistributedODEAlgorithm) = true
include("integrators.jl")


# Include concrete implementations
include("solvers/multirate.jl")
include("solvers/lsrk.jl")
include("solvers/ssprk.jl")
include("solvers/ark.jl")
include("solvers/ars.jl")
include("solvers/mis.jl")
include("solvers/wickerskamarock.jl")
include("solvers/rosenbrock.jl")

include("callbacks.jl")

end
