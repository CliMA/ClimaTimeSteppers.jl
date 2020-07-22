export AbstractIncrementingODEFunction, IncrementingODEFunction, IncrementingODEProblem

abstract type AbstractIncrementingODEFunction <: DiffEqBase.AbstractODEFunction{true}
end
struct IncrementingODEFunction{F} <: AbstractIncrementingODEFunction
    f::F
end
(f::IncrementingODEFunction)(args...; kwargs...) = f.f(args...; kwargs...)

const IncrementingODEProblem{uType,tType,P,F<: AbstractIncrementingODEFunction,K,PT} =
    DiffEqBase.ODEProblem{uType,tType,true,P,F,K,PT}

IncrementingODEProblem(f::AbstractIncrementingODEFunction, args...; kwargs...) = 
    DiffEqBase.ODEProblem(f, args...; kwargs...)

IncrementingODEProblem(f, args...; kwargs...) =
    DiffEqBase.ODEProblem(IncrementingODEFunction(f), args...; kwargs...)


