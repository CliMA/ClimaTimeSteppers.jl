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
using LinearOperators
using StaticArrays
using CUDA

array_device(::Union{Array, SArray, MArray}) = CPU()
array_device(::CuArray) = CUDADevice()
realview(x::Union{Array, SArray, MArray}) = x
realview(x::CuArray) = x


import DiffEqBase, SciMLBase, LinearAlgebra, DiffEqCallbacks, Krylov

include("sparse_containers.jl")
include("functions.jl")
include("operators.jl")
include("algorithms.jl")

abstract type DistributedODEAlgorithm <: DiffEqBase.AbstractODEAlgorithm end

abstract type AbstractIMEXARKAlgorithm <: DistributedODEAlgorithm end

abstract type AbstractTableau end
abstract type AbstractIMEXARKTableau <: AbstractTableau end

"""
    tableau(::DistributedODEAlgorithm)

Returns the tableau for a particular algorithm.
"""
function tableau end

"""
    theoretical_convergence_order

Returns the theoretical convergence order of an ODE algorithm
"""
function theoretical_convergence_order end
theoretical_convergence_order(tab) = error("No convergence order found for tableau $tab, please open an issue or PR.")

SciMLBase.allowscomplex(alg::DistributedODEAlgorithm) = true
include("integrators.jl")

include("solvers/update_signal_handler.jl")
include("solvers/convergence_condition.jl")
include("solvers/convergence_checker.jl")
include("solvers/newtons_method.jl")
include("solvers/imex_ark_tableaus.jl")
include("solvers/imex_ark.jl")

# Include concrete implementations
include("solvers/multirate.jl")
include("solvers/lsrk.jl")
include("solvers/ssprk.jl")
include("solvers/ark.jl")
# include("solvers/ars.jl") # previous implementations of ARS schemes
include("solvers/mis.jl")
include("solvers/wickerskamarock.jl")
include("solvers/rosenbrock.jl")

include("callbacks.jl")

include("convergence_orders.jl")

end
