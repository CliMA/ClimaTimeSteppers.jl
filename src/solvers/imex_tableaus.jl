export IMEXTableau, IMEXAlgorithm
export ARS111, ARS121, ARS122, ARS233, ARS232, ARS222, ARS343, ARS443
export IMKG232a, IMKG232b, IMKG242a, IMKG242b, IMKG243a, IMKG252a, IMKG252b
export IMKG253a, IMKG253b, IMKG254a, IMKG254b, IMKG254c, IMKG342a, IMKG343a
export DBM453, HOMMEM1
export SSP222, SSP322, SSP332, SSP333, SSP433

using StaticArrays: @SArray, SMatrix, sacollect

abstract type IMEXAlgorithmName <: AbstractAlgorithmName end
abstract type IMEXSSPRKAlgorithmName <: IMEXAlgorithmName end
default_constraint(::IMEXAlgorithmName) = Unconstrained()
default_constraint(::IMEXSSPRKAlgorithmName) = SSPConstrained()

"""
    IMEXTableau(; a_exp, b_exp, c_exp, a_imp, b_imp, c_imp)

A wrapper for an IMEX Butcher tableau (or, more accurately, a pair of Butcher
tableaus, one for explicit tendencies and the other for implicit tendencies).
Only `a_exp` and `a_imp` are required arguments; the default values for `b_exp`
and `b_imp` assume that the algorithm is FSAL (first same as last), and the
default values for `c_exp` and `c_imp` assume that it is internally consistent.

The explicit tableau must be strictly lower triangular, and the implicit tableau
must be lower triangular (only DIRK algorithms are currently supported).
"""
struct IMEXTableau{VS <: StaticArrays.StaticArray, MS <: StaticArrays.StaticArray}
    a_exp::MS # matrix of size s×s
    b_exp::VS # vector of length s
    c_exp::VS # vector of length s
    a_imp::MS # matrix of size s×s
    b_imp::VS # vector of length s
    c_imp::VS # vector of length s
end
function IMEXTableau(;
    a_exp,
    b_exp = a_exp[end, :],
    c_exp = vec(sum(a_exp; dims = 2)),
    a_imp,
    b_imp = a_imp[end, :],
    c_imp = vec(sum(a_imp; dims = 2)),
)
    @assert all(iszero, UpperTriangular(a_exp))
    @assert all(iszero, UpperTriangular(a_imp) - Diagonal(a_imp))

    # TODO: add generic promote_eltype
    a_exp, a_imp = promote(a_exp, a_imp)
    b_exp, b_imp, c_exp, c_imp = promote(b_exp, b_imp, c_exp, c_imp)
    return IMEXTableau(a_exp, b_exp, c_exp, a_imp, b_imp, c_imp)
end

"""
    IMEXAlgorithm(tableau, newtons_method, [constraint])
    IMEXAlgorithm(name, newtons_method, [constraint])
    [Name](newtons_method)

Constructs an IMEX algorithm for solving ODEs, with an optional name and
constraint. The first constructor accepts any `IMEXTableau` and an optional
constraint, leaving the algorithm unnamed. The second constructor automatically
determines the tableau and the default constraint from the algorithm name, which
must be an `IMEXAlgorithmName`.

The last constructor matches the notation of `OrdinaryDiffEq.jl`; it dispatches
to the second constructor by returning `IMEXAlgorithm(Name(), newtons_method)`.
"""
struct IMEXAlgorithm{
    C <: AbstractAlgorithmConstraint,
    N <: Union{Nothing, IMEXAlgorithmName},
    T <: IMEXTableau,
    NM <: NewtonsMethod,
} <: DistributedODEAlgorithm
    constraint::C
    name::N
    tableau::T
    newtons_method::NM
end
IMEXAlgorithm(tableau::IMEXTableau, newtons_method, constraint = Unconstrained()) =
    IMEXAlgorithm(constraint, nothing, tableau, newtons_method)
IMEXAlgorithm(name::IMEXAlgorithmName, newtons_method, constraint = default_constraint(name)) =
    IMEXAlgorithm(constraint, name, IMEXTableau(name), newtons_method)
(::Type{Name})(newtons_method::NewtonsMethod) where {Name <: IMEXAlgorithmName} = IMEXAlgorithm(Name(), newtons_method)

################################################################################

# ARS algorithms

# From Section 2 of "Implicit-Explicit Runge-Kutta Methods for Time-Dependent
# Partial Differential Equations" by Ascher et al.

# Each algorithm is named ARSsσp, where s is the number of implicit stages, σ is
# the number of explicit stages, and p is the order of accuracy

# This algorithm is equivalent to OrdinaryDiffEq.IMEXEuler.

"""
    ARS111

The Forward-Backward (1,1,1) implicit-explicit (IMEX) Runge-Kutta scheme of
[ARS1997](@cite), section 2.1.

This is equivalent to the `OrdinaryDiffEq.IMEXEuler` algorithm.
"""
struct ARS111 <: IMEXAlgorithmName end

function IMEXTableau(::ARS111)
    IMEXTableau(; a_exp = @SArray([0 0; 1 0]), a_imp = @SArray([0 0; 0 1]))
end

"""
    ARS121

The Forward-Backward (1,2,1) implicit-explicit (IMEX) Runge-Kutta scheme of
[ARS1997](@cite), section 2.2.

This is equivalent to the `OrdinaryDiffEq.IMEXEulerARK` algorithm.
"""
struct ARS121 <: IMEXAlgorithmName end

function IMEXTableau(::ARS121)
    IMEXTableau(; a_exp = @SArray([0 0; 1 0]), b_exp = @SArray([0, 1]), a_imp = @SArray([0 0; 0 1]))
end

struct ARS122 <: IMEXAlgorithmName end
function IMEXTableau(::ARS122)
    IMEXTableau(;
        a_exp = @SArray([0 0; 1/2 0]),
        b_exp = @SArray([0, 1]),
        a_imp = @SArray([0 0; 0 1/2]),
        b_imp = @SArray([0, 1]),
    )
end

struct ARS233 <: IMEXAlgorithmName end
function IMEXTableau(::ARS233)
    γ = 1 / 2 + √3 / 6
    IMEXTableau(;
        a_exp = @SArray([
            0 0 0
            γ 0 0
            (γ-1) (2-2γ) 0
        ]),
        b_exp = @SArray([0, 1 / 2, 1 / 2]),
        a_imp = @SArray([
            0 0 0
            0 γ 0
            0 (1-2γ) γ
        ]),
        b_imp = @SArray([0, 1 / 2, 1 / 2]),
    )
end

"""
    ARS232

The Forward-Backward (2,3,2) implicit-explicit (IMEX) Runge-Kutta scheme of
[ARS1997](@cite), section 2.5.
"""
struct ARS232 <: IMEXAlgorithmName end
function IMEXTableau(::ARS232)
    γ = 1 - √2 / 2
    δ = -2√2 / 3
    IMEXTableau(;
        a_exp = @SArray([
            0 0 0
            γ 0 0
            δ (1-δ) 0
        ]),
        b_exp = @SArray([0, 1 - γ, γ]),
        a_imp = @SArray([
            0 0 0
            0 γ 0
            0 (1-γ) γ
        ]),
    )
end

struct ARS222 <: IMEXAlgorithmName end
function IMEXTableau(::ARS222)
    γ = 1 - √2 / 2
    δ = 1 - 1 / 2γ
    IMEXTableau(; a_exp = @SArray([
        0 0 0
        γ 0 0
        δ (1-δ) 0
    ]), a_imp = @SArray([
        0 0 0
        0 γ 0
        0 (1-γ) γ
    ]))
end

"""
    ARS343

The L-stable, third-order (3,4,3) implicit-explicit (IMEX) Runge-Kutta scheme of
[ARS1997](@cite), section 2.7.
"""
struct ARS343 <: IMEXAlgorithmName end
function IMEXTableau(::ARS343)
    γ = 0.4358665215084590
    a42 = 0.5529291480359398
    a43 = a42
    b1 = -3 / 2 * γ^2 + 4 * γ - 1 / 4
    b2 = 3 / 2 * γ^2 - 5 * γ + 5 / 4
    a31 =
        (1 - 9 / 2 * γ + 3 / 2 * γ^2) * a42 + (11 / 4 - 21 / 2 * γ + 15 / 4 * γ^2) * a43 - 7 / 2 + 13 * γ - 9 / 2 * γ^2
    a32 =
        (-1 + 9 / 2 * γ - 3 / 2 * γ^2) * a42 + (-11 / 4 + 21 / 2 * γ - 15 / 4 * γ^2) * a43 + 4 - 25 / 2 * γ +
        9 / 2 * γ^2
    a41 = 1 - a42 - a43
    IMEXTableau(;
        a_exp = @SArray([
            0 0 0 0
            γ 0 0 0
            a31 a32 0 0
            a41 a42 a43 0
        ]),
        b_exp = @SArray([0, b1, b2, γ]),
        a_imp = @SArray([
            0 0 0 0
            0 γ 0 0
            0 (1 - γ)/2 γ 0
            0 b1 b2 γ
        ]),
    )
end

struct ARS443 <: IMEXAlgorithmName end
function IMEXTableau(::ARS443)
    IMEXTableau(;
        a_exp = @SArray([
            0 0 0 0 0
            1/2 0 0 0 0
            11/18 1/18 0 0 0
            5/6 -5/6 1/2 0 0
            1/4 7/4 3/4 -7/4 0
        ]),
        a_imp = @SArray([
            0 0 0 0 0
            0 1/2 0 0 0
            0 1/6 1/2 0 0
            0 -1/2 1/2 1/2 0
            0 3/2 -3/2 1/2 1/2
        ]),
    )
end

################################################################################

# IMKG algorithms

# From Tables 3 and 4 of "Efficient IMEX Runge-Kutta Methods for Nonhydrostatic
# Dynamics" by Steyer et al.

# Each algorithm is named IMKGpfjl, where p is the order of accuracy, f is the
# number of explicit stages, j is the number of implicit stages, and l is an
# identifying letter

# TODO: Tables 3 and 4 are riddled with typos, but most of these can be easily
# identified and corrected by referencing the implementations in HOMME:
# https://github.com/E3SM-Project/E3SM/blob/master/components/homme/src/arkode/arkode_tables.F90
# Unfortunately, the implementations of IMKG353a and IMKG354a in HOMME also
# appear to be wrong, so they are not included here. Eventually, we should get
# the official implementations from the paper's authors.

# a_exp:                                   a_imp:
# 0      0      ⋯      0      0      0     0      0      ⋯      0      0      0
# α[1]   0      ⋱      ⋮      ⋮      ⋮     α̂[1]   δ̂[1]   ⋱      ⋮      ⋮       0
# β[1]   α[2]   ⋱      0      0      0    β[1]   α̂[2]   ⋱      0      0      0
# ⋮      0      ⋱      0      0      0     ⋮      0      ⋱      δ̂[s-3] 0      0
# ⋮      ⋮       ⋱     α[s-2] 0      0     ⋮      ⋮       ⋱      α̂[s-2] δ̂[s-2] 0
# β[s-2] 0      ⋯      0      α[s-1] 0     β[s-2] 0      ⋯       0      α̂[s-1] 0
imkg_exp(i, j, α, β) = i == j + 1 ? α[j] : (i > 2 && j == 1 ? β[i - 2] : 0)
imkg_imp(i, j, α̂, β, δ̂) =
    i == j + 1 ? α̂[j] : (i > 2 && j == 1 ? β[i - 2] : (1 < i <= length(α̂) && i == j ? δ̂[i - 1] : 0))
function IMKGTableau(α, α̂, δ̂, β = ntuple(_ -> 0, length(δ̂)))
    s = length(α̂) + 1
    type = SMatrix{s, s}
    return IMEXTableau(;
        a_exp = sacollect(type, imkg_exp(i, j, α, β) for i in 1:s, j in 1:s),
        a_imp = sacollect(type, imkg_imp(i, j, α̂, β, δ̂) for i in 1:s, j in 1:s),
    )
end

struct IMKG232a <: IMEXAlgorithmName end
function IMEXTableau(::IMKG232a)
    IMKGTableau((1 / 2, 1 / 2, 1), (0, -1 / 2 + √2 / 2, 1), (1 - √2 / 2, 1 - √2 / 2))
end

struct IMKG232b <: IMEXAlgorithmName end
function IMEXTableau(::IMKG232b)
    IMKGTableau((1 / 2, 1 / 2, 1), (0, -1 / 2 - √2 / 2, 1), (1 + √2 / 2, 1 + √2 / 2))
end

struct IMKG242a <: IMEXAlgorithmName end
function IMEXTableau(::IMKG242a)
    IMKGTableau((1 / 4, 1 / 3, 1 / 2, 1), (0, 0, -1 / 2 + √2 / 2, 1), (0, 1 - √2 / 2, 1 - √2 / 2))
end

struct IMKG242b <: IMEXAlgorithmName end
function IMEXTableau(::IMKG242b)
    IMKGTableau((1 / 4, 1 / 3, 1 / 2, 1), (0, 0, -1 / 2 - √2 / 2, 1), (0, 1 + √2 / 2, 1 + √2 / 2))
end

# The paper uses √3/6 for α̂[3], which also seems to work.
struct IMKG243a <: IMEXAlgorithmName end
function IMEXTableau(::IMKG243a)
    IMKGTableau((1 / 4, 1 / 3, 1 / 2, 1), (0, 1 / 6, -√3 / 6, 1), (1 / 2 + √3 / 6, 1 / 2 + √3 / 6, 1 / 2 + √3 / 6))
end

struct IMKG252a <: IMEXAlgorithmName end
function IMEXTableau(::IMKG252a)
    IMKGTableau((1 / 4, 1 / 6, 3 / 8, 1 / 2, 1), (0, 0, 0, -1 / 2 + √2 / 2, 1), (0, 0, 1 - √2 / 2, 1 - √2 / 2))
end

struct IMKG252b <: IMEXAlgorithmName end
function IMEXTableau(::IMKG252b)
    IMKGTableau((1 / 4, 1 / 6, 3 / 8, 1 / 2, 1), (0, 0, 0, -1 / 2 - √2 / 2, 1), (0, 0, 1 + √2 / 2, 1 + √2 / 2))
end

# The paper uses 0.08931639747704086 for α̂[3], which also seems to work.
struct IMKG253a <: IMEXAlgorithmName end
function IMEXTableau(::IMKG253a)
    IMKGTableau(
        (1 / 4, 1 / 6, 3 / 8, 1 / 2, 1),
        (0, 0, √3 / 4 * (1 - √3 / 3) * ((1 + √3 / 3)^2 - 2), √3 / 6, 1),
        (0, 1 / 2 - √3 / 6, 1 / 2 - √3 / 6, 1 / 2 - √3 / 6),
    )
end

# The paper uses 1.2440169358562922 for α̂[3], which also seems to work.
struct IMKG253b <: IMEXAlgorithmName end
function IMEXTableau(::IMKG253b)
    IMKGTableau(
        (1 / 4, 1 / 6, 3 / 8, 1 / 2, 1),
        (0, 0, √3 / 4 * (1 + √3 / 3) * ((1 - √3 / 3)^2 - 2), -√3 / 6, 1),
        (0, 1 / 2 + √3 / 6, 1 / 2 + √3 / 6, 1 / 2 + √3 / 6),
    )
end

struct IMKG254a <: IMEXAlgorithmName end
function IMEXTableau(::IMKG254a)
    IMKGTableau((1 / 4, 1 / 6, 3 / 8, 1 / 2, 1), (0, -3 / 10, 5 / 6, -3 / 2, 1), (-1 / 2, 1, 1, 2))
end

struct IMKG254b <: IMEXAlgorithmName end
function IMEXTableau(::IMKG254b)
    IMKGTableau((1 / 4, 1 / 6, 3 / 8, 1 / 2, 1), (0, -1 / 20, 5 / 4, -1 / 2, 1), (-1 / 2, 1, 1, 1))
end

struct IMKG254c <: IMEXAlgorithmName end
function IMEXTableau(::IMKG254c)
    IMKGTableau((1 / 4, 1 / 6, 3 / 8, 1 / 2, 1), (0, 1 / 20, 5 / 36, 1 / 3, 1), (1 / 6, 1 / 6, 1 / 6, 1 / 6))
end

# The paper and HOMME completely disagree on this algorithm. Since the version
# in the paper is not "342" (it appears to be "332"), the version from HOMME is
# used here.
# const IMKG342a = IMKGTableau(
#     (0, 1/3, 1/3, 3/4),
#     (0, -1/6 - √3/6, -1/6 - √3/6, 3/4),
#     (0, 1/2 + √3/6, 1/2 + √3/6),
#     (1/3, 1/3, 1/4),
# )
struct IMKG342a <: IMEXAlgorithmName end
function IMEXTableau(::IMKG342a)
    IMKGTableau(
        (1 / 4, 2 / 3, 1 / 3, 3 / 4),
        (0, 1 / 6 - √3 / 6, -1 / 6 - √3 / 6, 3 / 4),
        (0, 1 / 2 + √3 / 6, 1 / 2 + √3 / 6),
        (0, 1 / 3, 1 / 4),
    )
end

struct IMKG343a <: IMEXAlgorithmName end
function IMEXTableau(::IMKG343a)
    IMKGTableau((1 / 4, 2 / 3, 1 / 3, 3 / 4), (0, -1 / 3, -2 / 3, 3 / 4), (-1 / 3, 1, 1), (0, 1 / 3, 1 / 4))
end

# The paper and HOMME completely disagree on this algorithm, but neither version
# is "353" (they appear to be "343" and "354", respectively).
# struct IMKG353a <: IMEXAlgorithmName end
# function IMEXTableau(::IMKG353a)
#     IMKGTableau(
#         (1/4, 2/3, 1/3, 3/4),
#         (0, -359/600, -559/600, 3/4),
#         (-1.1678009811335388, 253/200, 253/200),
#         (0, 1/3, 1/4),
#     )
# end
# struct IMKG353a <: IMEXAlgorithmName end
# function IMEXTableau(::IMKG353a)
#     IMKGTableau(
#         (-0.017391304347826087, -23/25, 5/3, 1/3, 3/4),
#         (0.3075640504095504, -1.2990164859879263, 751/600, -49/60, 3/4),
#         (-0.2981612530370581, 83/200, 83/200, 23/20),
#         (1, -1, 1/3, 1/4),
#     )
# end

# The version of this algorithm in the paper is not "354" (it appears to be
# "253"), and this algorithm is missing from HOMME (or, more precisely, the
# tableau for IMKG353a is mistakenly used to define IMKG354a, and the tableau
# for IMKG354a is not specified).
# struct IMKG354a <: IMEXAlgorithmName end
# function IMEXTableau(::IMKG354a)
#     IMKGTableau(
#         (1/5, 1/5, 2/3, 1/3, 3/4),
#         (0, 0, 11/30, -2/3, 3/4),
#         (0, 2/4, 2/5, 1),
#         (0, 0, 1/3, 1/4),
#     )
# end

################################################################################

# DBM algorithm

# From Appendix A of "Evaluation of Implicit-Explicit Additive Runge-Kutta
# Integrators for the HOMME-NH Dynamical Core" by Vogl et al.

# The algorithm has 4 implicit stages, 5 overall stages, and 3rd order accuracy.

struct DBM453 <: IMEXAlgorithmName end
function IMEXTableau(::DBM453)
    γ = 0.32591194130117247
    IMEXTableau(;
        a_exp = @SArray(
            [
                0 0 0 0 0
                0.10306208811591838 0 0 0 0
                -0.94124866143519894 1.6626399742527356 0 0 0
                -1.3670975201437765 1.3815852911016873 1.2673234025619065 0 0
                -0.81287582068772448 0.81223739060505738 0.90644429603699305 0.094194134045674111 0
            ]
        ),
        b_exp = @SArray([0.87795339639076675, -0.72692641526151547, 0.7520413715737272, -0.22898029400415088, γ]),
        a_imp = @SArray(
            [
                0 0 0 0 0
                -0.2228498531852541 γ 0 0 0
                -0.46801347074080545 0.86349284225716961 γ 0 0
                -0.46509906651927421 0.81063103116959553 0.61036726756832357 γ 0
                0.87795339639076675 -0.72692641526151547 0.7520413715737272 -0.22898029400415088 γ
            ]
        ),
    )
end

################################################################################

# HOMMEM1 algorithm

# From Section 4.1 of "A framework to evaluate IMEX schemes for atmospheric
# models" by Guba et al.

# The algorithm has 5 implicit stages, 6 overall stages, and 2rd order accuracy.

struct HOMMEM1 <: IMEXAlgorithmName end
function IMEXTableau(::HOMMEM1)
    IMEXTableau(;
        a_exp = @SArray([
            0 0 0 0 0 0
            1/5 0 0 0 0 0
            0 1/5 0 0 0 0
            0 0 1/3 0 0 0
            0 0 0 1/2 0 0
            0 0 0 0 1 0
        ]),
        a_imp = @SArray([
            0 0 0 0 0 0
            0 1/5 0 0 0 0
            0 0 1/5 0 0 0
            0 0 0 1/3 0 0
            0 0 0 0 1/2 0
            5/18 5/18 0 0 0 8/18
        ]),
    )
end

################################################################################

# IMEX SSP algorithms

"""
    SSP222

https://link.springer.com/content/pdf/10.1007/BF02728986.pdf, Table II
"""
struct SSP222 <: IMEXSSPRKAlgorithmName end
function IMEXTableau(::SSP222)
    γ = 1 - √2 / 2
    return IMEXTableau(;
        a_exp = @SArray([
            0 0
            1 0
        ]),
        b_exp = @SArray([1 / 2, 1 / 2]),
        a_imp = @SArray([
            γ 0
            (1-2γ) γ
        ]),
        b_imp = @SArray([1 / 2, 1 / 2]),
    )
end

"""
    SSP322

https://link.springer.com/content/pdf/10.1007/BF02728986.pdf, Table III
"""
struct SSP322 <: IMEXSSPRKAlgorithmName end
function IMEXTableau(::SSP322)
    return IMEXTableau(;
        a_exp = @SArray([
            0 0 0
            0 0 0
            0 1 0
        ]),
        b_exp = @SArray([0, 1 / 2, 1 / 2]),
        a_imp = @SArray([
            1/2 0 0
            -1/2 1/2 0
            0 1/2 1/2
        ]),
        b_imp = @SArray([0, 1 / 2, 1 / 2]),
    )
end

"""
    SSP332

https://link.springer.com/content/pdf/10.1007/BF02728986.pdf, Table V
"""
struct SSP332 <: IMEXSSPRKAlgorithmName end
function IMEXTableau(::SSP332)
    γ = 1 - √2 / 2
    return IMEXTableau(;
        a_exp = @SArray([
            0 0 0
            1 0 0
            1/4 1/4 0
        ]),
        b_exp = @SArray([1 / 6, 1 / 6, 2 / 3]),
        a_imp = @SArray([
            γ 0 0
            (1-2γ) γ 0
            (1 / 2-γ) 0 γ
        ]),
        b_imp = @SArray([1 / 6, 1 / 6, 2 / 3]),
    )
end

"""
    SSP333([β])

Family of SSP333 algorithms parametrized by the value β, from Section 3.2 of
https://arxiv.org/pdf/1702.04621.pdf. The default value of β, 1/2 + √3/6,
results in an SDIRK algorithm, which is also called SSP3(333)c in
https://gmd.copernicus.org/articles/11/1497/2018/gmd-11-1497-2018.pdf.
"""
struct SSP333{FT} <: IMEXSSPRKAlgorithmName
    β::FT
    SSP333(β::AbstractFloat = 1 / 2 + √3 / 6) = new{typeof(β)}(β)
end
function IMEXTableau((; β)::SSP333)
    @assert β > 1 / 2
    γ = (2β^2 - 3β / 2 + 1 / 3) / (2 - 4β)
    return IMEXTableau(;
        a_exp = @SArray([
            0 0 0
            1 0 0
            1/4 1/4 0
        ]),
        b_exp = @SArray([1 / 6, 1 / 6, 2 / 3]),
        a_imp = @SArray([
            0 0 0
            (4γ+2β) (1 - 4γ-2β) 0
            (1 / 2 - β-γ) γ β
        ]),
        b_imp = @SArray([1 / 6, 1 / 6, 2 / 3]),
    )
end

"""
    SSP433

https://link.springer.com/content/pdf/10.1007/BF02728986.pdf, Table VI
"""
struct SSP433 <: IMEXSSPRKAlgorithmName end
function IMEXTableau(::SSP433)
    α = 0.24169426078821
    β = 0.06042356519705
    η = 0.12915286960590
    return IMEXTableau(;
        a_exp = @SArray([
            0 0 0 0
            0 0 0 0
            0 1 0 0
            0 1/4 1/4 0
        ]),
        b_exp = @SArray([0, 1 / 6, 1 / 6, 2 / 3]),
        a_imp = @SArray([
            α 0 0 0
            -α α 0 0
            0 (1-α) α 0
            β η (1 / 2 - α - β-η) α
        ]),
        b_imp = @SArray([0, 1 / 6, 1 / 6, 2 / 3]),
    )
end
