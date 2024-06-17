export SSP22Heuns, SSP33ShuOsher, RK4

is_strictly_lower_triangular(matrix) = all(iszero, UpperTriangular(matrix))
is_lower_triangular(matrix) = all(iszero, UpperTriangular(matrix) - Diagonal(matrix))

"""
    RKTableau(a, b, c, d, Γ, α)

A container for all of the information required to formulate a Runge-Kutta (RK)
timestepping method. The arrays `a`, `b`, and `c` comprise the Butcher tableau
of the method, while `α` is either `nothing` or the first matrix of the method's
canonical Shu-Osher formulation (if at least one such formulation is available).

`Γ` is an additional tableau that can be provided. For purely RK schemes, this
is just the diagonal of `a`. For Rosenbrock schemes, it is a matrix with the
same shape as `a`.

`d` is also an additional time-related tableau. For purely RK schemes, this is a
copy of `c`. For Rosenbrock schemes, this is typically used for evaluating
explicit time derivatives.
"""
struct RKTableau{FT, AN <: Union{Nothing, Array{FT, 2}}}
    a::Array{FT, 2}
    b::Array{FT, 1}
    c::Array{FT, 1}
    d::Array{FT, 1}
    Γ::Array{FT, 2}
    α::AN
end
function RKTableau(a, b, c, d, Γ, α)
    s = length(b)
    size(a) == (s, s) || error("invalid Butcher tableau matrix")
    size(Γ) == (s, s) || error("invalid Γ tableau matrix")
    size(b) == size(c) == size(d) == (s,) || error("invalid Butcher tableau vector")

    is_lower_triangular(a) || error("Butcher tableau matrix is not ERK or DIRK")
    is_lower_triangular(Γ) || error("Γ tableau matrix is not lower triangular")

    sum(b) == 1 || @warn "tableau does not obey 1st order consistency condition"
    vec(sum(a; dims = 2)) == c || @warn "tableau is not internally consistent"

    if !isnothing(α)
        size(α) == (s + 1, s) || error("invalid Shu-Osher form matrix")

        is_lower_triangular(α[1:s, :]) || error("Shu-Osher form matrix is not ERK or DIRK")
        if is_strictly_lower_triangular(a)
            is_strictly_lower_triangular(α[1:s, :]) ||
                error("Shu-Osher form matrix is DIRK while Butcher tableau matrix is ERK")
        end

        vec(sum(α; dims = 2)) in (vcat([0], ones(s)), vcat([0, 0], ones(s - 1))) ||
            error("Shu-Osher form is not canonical")

        # TODO: Add support for applying limiters in reverse (negative coefficients in α).
        all(>=(0), α) || error("Shu-Osher form matrix has negative coefficients")
    end

    FT = Base.promote_eltype(a, b, c, d, Γ, (isnothing(α) ? () : (α,))...)
    return RKTableau{FT, typeof(α)}(a, b, c, d, Γ, α)
end

Base.eltype(::Type{RKTableau{FT}}) where {FT} = FT

Base.promote_rule(::Type{<:RKTableau{FT1}}, ::Type{<:RKTableau{FT2}}) where {FT1, FT2} =
    RKTableau{promote_type(FT1, FT2)}

function Base.convert(::Type{RKTableau{FT}}, tableau::RKTableau) where {FT}
    (; a, b, c, d, Γ, α) = tableau
    return RKTableau{FT, isnothing(α) ? Nothing : Array{FT, 2}}(a, b, c, d, Γ, α)
end

"""
    ButcherTableau(a, [b], [c], [d], [Γ])

Constructs an `RKTableau` without a Shu-Osher formulation, under the default
assumptions that it is first-same-as-last (FSAL) and internally consistent.
"""
ButcherTableau(a, b = a[end, :], c = vec(sum(a; dims = 2)), d = copy(c), Γ = diagm(diag(a))) = RKTableau(a, b, c, d, Γ, nothing)

"""
    RosenbrockTableau(a_square, b, Γ)

Constructs an `RKTableau` with a Rosenbrock tableau.
"""
RosenbrockTableau(Γ, a, b = a[end, :], c = vec(sum(a; dims = 2)), d = vec(sum(Γ; dims = 2))) = RKTableau(a, b, c, d, Γ, nothing)

"""
    ShuOsherTableau(α, a, [b], [c], [d], [Γ])

Constructs an `RKTableau` with a canonical Shu-Osher formulation whose first
matrix is given by `α`, under the default assumptions that it is
first-same-as-last (FSAL) and internally consistent.
"""
ShuOsherTableau(α, a, b = a[end, :], c = vec(sum(a; dims = 2)), d = copy(c), Γ = diagm(diag(a))) = RKTableau(a, b, c, d, Γ, α)

"""
    PaddedTableau(tableau)

Constructs an `RKTableau` that is identical to the given `RKTableau`, but with
an additional "empty" stage at the beginning of the method that leaves the
initial state unmodified.
"""
function PaddedTableau(tableau)
    s = length(tableau.b)
    return RKTableau(
        vcat(zeros(s + 1)', hcat(zeros(s), tableau.a)),
        vcat([0], tableau.b),
        vcat([0], tableau.c),
        vcat([0], tableau.d),
        vcat(zeros(s + 1)', hcat(zeros(s), tableau.Γ)),
        isnothing(tableau.α) ? nothing : vcat(zeros(s + 1)', hcat(zeros(s + 1), tableau.α)),
    )
end

"""
    is_ERK(tableau)

Checks whether an `RKTableau` is explicit; i.e., whether its coefficient
matrices are strictly lower triangular.
"""
is_ERK(tableau) = is_strictly_lower_triangular(tableau.a)

"""
    is_DIRK(tableau)

Checks whether an `RKTableau` is diagonally implicit; i.e., whether its
coefficient matrices are non-strictly lower triangular.
"""
is_DIRK(tableau) = is_lower_triangular(tableau.a) && !is_strictly_lower_triangular(tableau.a)

"""
    RKAlgorithmName

An `AbstractAlgorithmName` with a method of the form `RKTableau(name)`.
"""
abstract type RKAlgorithmName <: AbstractAlgorithmName end

"""
    SSPRKAlgorithmName

An `RKAlgorithmName` whose tableau has a canonical Shu-Osher formulation.
"""
abstract type SSPRKAlgorithmName <: RKAlgorithmName end

################################################################################

"""
    SSP22Heuns

An SSPRK algorithm from [SO1988](@cite), with 2 stages and 2nd order accuracy.
Also called Heun's method ([Heun1900](@cite)).
"""
struct SSP22Heuns <: SSPRKAlgorithmName end
RKTableau(::SSP22Heuns) = ShuOsherTableau(
    [
        0 0
        1 0
        1//2 1//2
    ],
    [
        0 0
        1 0
    ],
    [1 // 2, 1 // 2],
)

"""
    SSP33ShuOsher

An SSPRK algorithm from [SO1988](@cite), with 3 stages and 3rd order accuracy.
"""
struct SSP33ShuOsher <: SSPRKAlgorithmName end
RKTableau(::SSP33ShuOsher) = ShuOsherTableau(
    [
        0 0 0
        1 0 0
        3//4 1//4 0
        1//3 0 2//3
    ],
    [
        0 0 0
        1 0 0
        1//4 1//4 0
    ],
    [1 // 6, 1 // 6, 2 // 3],
)

"""
    RK4

The RK4 algorithm from [SM2003](@cite), a Runge-Kutta method with
4 stages and 4th order accuracy.
"""
struct RK4 <: RKAlgorithmName end
RKTableau(::RK4) = ButcherTableau(
    [
        0 0 0 0
        1//2 0 0 0
        0 1//2 0 0
        0 0 1 0
    ],
    [1 // 6, 1 // 3, 1 // 3, 1 // 6],
)

abstract type RosenbrockAlgorithmName <: AbstractAlgorithmName end

"""
    SSPKnoth

`SSPKnoth` is a third-order Rosenbrock method developed by Oswald Knoth. When
integrating an implicit tendency, this reduces to a second-order method because
it only performs an approximate implicit solve on each stage.

The coefficients are the same as in `CGDycore.jl`, except that for C we add the
diagonal elements too. Note, however, that the elements on the diagonal of C do
not really matter because C is only used in its lower triangular part. We add them
mostly to match literature on the subject
"""
struct SSPKnoth <: RosenbrockAlgorithmName end

function RKTableau(::SSPKnoth)
    return RosenbrockTableau(
        [
            1 0 0
            0 1 0
            -3//4 -3//4 1
        ],
        [
            0 0 0
            1 0 0
            1//4 1//4 0
        ],
        [1 // 6, 1 // 6, 2 // 3],
    )
end
