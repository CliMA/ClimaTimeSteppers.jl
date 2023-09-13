##########
# Patch fix for https://github.com/CliMA/ClimaCore.jl/issues/1453
# Remove once https://github.com/CliMA/ClimaCore.jl/pull/1454
# is merged
import ClimaCore.RecursiveApply as RA
RA.rmap(fn::F, X) where {F} = fn(X)
RA.rmap(fn::F, X::Tuple{}) where {F} = ()
RA.rmap(fn::F, X::Tuple) where {F} = (RA.rmap(fn, first(X)), RA.rmap(fn, Base.tail(X))...)
RA.rmap(fn::F, X::NamedTuple{names}) where {F, names} = NamedTuple{names}(RA.rmap(fn, Tuple(X)))

RA.rmap(fn::F, X, Y) where {F} = fn(X, Y)
RA.rmap(fn::F, X::Tuple{}, Y::Tuple{}) where {F} = ()
RA.rmap(fn::F, X::Tuple{}, Y) where {F} = ()
RA.rmap(fn::F, X, Y::Tuple{}) where {F} = ()
RA.rmap(fn::F, X::Tuple, Y::Tuple) where {F} =
    (RA.rmap(fn, first(X), first(Y)), RA.rmap(fn, Base.tail(X), Base.tail(Y))...)

RA.rmap(fn::F, X, Y::Tuple) where {F} = (RA.rmap(fn, X, first(Y)), RA.rmap(fn, X, Base.tail(Y))...)

RA.rmap(fn::F, X::Tuple, Y) where {F} = (RA.rmap(fn, first(X), Y), RA.rmap(fn, Base.tail(X), Y)...)

RA.rmap(fn::F, X::NamedTuple{names}, Y::NamedTuple{names}) where {F, names} =
    NamedTuple{names}(RA.rmap(fn, Tuple(X), Tuple(Y)))
RA.rmap(fn::F, X::NamedTuple{names}, Y) where {F, names} = NamedTuple{names}(RA.rmap(fn, Tuple(X), Y))
RA.rmap(fn::F, X, Y::NamedTuple{names}) where {F, names} = NamedTuple{names}(RA.rmap(fn, X, Tuple(Y)))
##########
