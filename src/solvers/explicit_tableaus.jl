export ExplicitTableau, ExplicitAlgorithm
export SSP22Heuns, SSP33ShuOsher, RK4

abstract type ERKAlgorithmName <: AbstractAlgorithmName end

"""
    ExplicitTableau(; a, [b], [c])

An explicit Butcher tableau. Usually constructed indirectly via an algorithm
name such as [`SSP33ShuOsher`](@ref).

# Keyword Arguments
- `a`: Butcher matrix (strictly lower triangular, required)
- `b`: weights (default: last row of `a`, i.e., first same as last, FSAL)
- `c`: abscissae (default: row sums of `a`)
"""
struct ExplicitTableau{A <: SPCO, B <: SPCO, C <: SPCO}
    a::A # matrix of size s×s
    b::B # vector of length s
    c::C # vector of length s
end
ExplicitTableau(args...) = ExplicitTableau(map(x -> SparseCoeffs(x), args)...)
function ExplicitTableau(; a, b = a[end, :], c = vec(sum(a; dims = 2)))
    @assert all(iszero, UpperTriangular(a))
    b, c = promote(b, c) # TODO: add generic promote_eltype
    return ExplicitTableau(a, b, c)
end

"""
    ExplicitAlgorithm(name, [constraint])
    ExplicitAlgorithm(tableau, [constraint])

An explicit Runge-Kutta algorithm. Shorthand for an [`IMEXAlgorithm`](@ref)
with identical explicit/implicit tableaux and no Newton solver.

# Arguments
- `name`: an algorithm name such as `SSP33ShuOsher()` or `RK4()`
- `constraint`: [`Unconstrained`](@ref) (default) or [`SSP`](@ref)

# Example
```julia
alg = ExplicitAlgorithm(SSP33ShuOsher())
```
"""
ExplicitAlgorithm(tableau::ExplicitTableau, constraint = Unconstrained()) =
    IMEXAlgorithm(constraint, nothing, IMEXTableau(tableau), nothing)
ExplicitAlgorithm(name::ERKAlgorithmName, constraint = default_constraint(name)) =
    IMEXAlgorithm(constraint, name, IMEXTableau(name), nothing)

IMEXTableau(name::ERKAlgorithmName) = IMEXTableau(ExplicitTableau(name))
IMEXTableau((; a, b, c)::ExplicitTableau) = IMEXTableau(a, b, c, a, b, c)

################################################################################

abstract type SSPRKAlgorithmName <: ERKAlgorithmName end

default_constraint(::SSPRKAlgorithmName) = SSP()

"""
    SSP22Heuns

An SSPRK algorithm from [SO1988](@cite), with 2 stages and 2nd order accuracy.
Also called Heun's method ([Heun1900](@cite)).
"""
struct SSP22Heuns <: SSPRKAlgorithmName end
function ExplicitTableau(::SSP22Heuns)
    return ExplicitTableau(; a = @SArray([
            0 0
            1 0
        ]), b = @SArray([1 / 2, 1 / 2]))
end

"""
    SSP33ShuOsher

An SSPRK algorithm from [SO1988](@cite), with 3 stages and 3rd order accuracy.
"""
struct SSP33ShuOsher <: SSPRKAlgorithmName end
function ExplicitTableau(::SSP33ShuOsher)
    return ExplicitTableau(;
        a = @SArray([
            0 0 0
            1 0 0
            1/4 1/4 0
        ]),
        b = @SArray([1 / 6, 1 / 6, 2 / 3])
    )
end

"""
    RK4

The RK4 algorithm from [SM2003](@cite), a Runge-Kutta method with
4 stages and 4th order accuracy.
"""
struct RK4 <: ERKAlgorithmName end
function ExplicitTableau(::RK4)
    return ExplicitTableau(;
        a = @SArray([
            0 0 0 0
            1/2 0 0 0
            0 1/2 0 0
            0 0 1 0
        ]),
        b = @SArray([1 / 6, 1 / 3, 1 / 3, 1 / 6])
    )
end
