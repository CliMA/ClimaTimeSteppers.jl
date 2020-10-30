"""
    DiffEqLinearOperator(f; isconstant=false) :: DiffEqBase.AbstractDiffEqLinearOperator 

A wrapper around a function `f` that acts as a linear operator.

`isconstant=true` if `f` is time-invariant.
"""
struct DiffEqLinearOperator{T,F,isconstant} <: DiffEqBase.AbstractDiffEqLinearOperator{T}
  f::F
end
DiffEqLinearOperator{T}(f; isconstant=false) where {T} = DiffEqLinearOperator{T,typeof(f), isconst}(f)

DiffEqBase.isconstant(::DiffEqLinearOperator{F,isconstant}) where {F,isconstant} = isconstant
(op::DiffEqLinearOperator)(args...) = op.f(args...)

"""
    EulerOperator(op, γ) :: DiffEqBase.AbstractDiffEqLinearOperator 

A linear operator which performs an explicit Euler step ``u + α f(u)``

, where `f!` and `op!` both operate inplace, with extra arguments passed
through, i.e.
```
op!(LQ, Q, args...)
```
is equivalent to
```
f!(dQ, Q, args...)
LQ .= Q .+ γ .* dQ
```
"""
mutable struct EulerOperator{T,F,P,tType}
    f::F
    γ::T
    p::P
    t::tType
end

function (op::EulerOperator)(LQ, Q, args...)
    op.f(LQ, Q, args...)
    @. LQ = Q + op.γ * LQ
end
DiffEqBase.isconstant(op::EulerOperator) = op.f isa DiffEqBase.AbstractDiffEqLinearOperator && DiffEqBase.isconstant(op.f)
LinearAlgebra.mul!(Y, op::EulerOperator, X) = op(Y,X,op.p,op.t)
