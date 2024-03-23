"""
    ClimaTimeSteppers

Ordinary differential equation solvers

JuliaDiffEq terminology:

* _Function_: the right-hand side function df/dt.
  * by default, a function gets wrapped in an `ODEFunction`
  * define new `IncrementingODEFunction` to support incrementing function calls.

* _Problem_: Function, initial u, time span, parameters and options

  `du/dt = f(u,p,t) = fL(u,p,t)  + fR(u,p,t)`

  `fR(u,p,t) == f(u.p,t) - fL(u,p,t)`
  `fL(u,_,_) == A*u for some `A` (matrix free)`

  SplitODEProlem(fL, fR)


  * `ODEProblem` from SciMLBase.jl
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
using LinearAlgebra
using LinearOperators
using StaticArrays
using UnrolledUtilities
import ClimaComms
using Colors
using NVTX

export AbstractAlgorithmName

array_device(::Union{Array, SArray, MArray}) = CPU()
array_device(x) = CUDADevice() # assume CUDA

float_type(::Type{T}) where {T} = T <: AbstractFloat ? T : promote_type(map(float_type, fieldtypes(T))...)

import DiffEqBase, SciMLBase, LinearAlgebra, DiffEqCallbacks, Krylov

include(joinpath("utilities", "sparse_tuple.jl"))
include("functions.jl")

abstract type DistributedODEAlgorithm <: DiffEqBase.AbstractODEAlgorithm end
SciMLBase.allowscomplex(alg::DistributedODEAlgorithm) = true
include("integrators.jl")

include("utilities/update_signal_handler.jl")
include("utilities/convergence_condition.jl")
include("utilities/convergence_checker.jl")
include("nl_solvers/newtons_method.jl")

"""
    AbstractAlgorithmName

Supertype of predefined Runge-Kutta methods.
"""
abstract type AbstractAlgorithmName end
include("solvers/rk_tableaus.jl")
include("solvers/ark_tableaus.jl")
include("solvers/ark_algorithm.jl")

include("solvers/multirate.jl")
include("solvers/lsrk.jl")
include("solvers/mis.jl")
include("solvers/wickerskamarock.jl")
include("solvers/rosenbrock.jl")

include("Callbacks.jl")


benchmark_step(integrator, device) =
    @warn "Must load CUDA, BenchmarkTools, OrderedCollections, StatsBase, PrettyTables to trigger the ClimaTimeSteppersBenchmarkToolsExt extension"

end
