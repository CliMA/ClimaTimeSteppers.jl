export ARS111, ARS121, ARS122, ARS233, ARS232, ARS222, ARS343, ARS443
export IMKG232a, IMKG232b, IMKG242a, IMKG242b, IMKG252a, IMKG252b
export IMKG253a, IMKG253b, IMKG254a, IMKG254b, IMKG254c, IMKG342a, IMKG343a
export DBM453, HOMMEM1

using StaticArrays: @SArray, SMatrix, sacollect

################################################################################

"""
    IMEXARKTableau(; a_exp, b_exp, c_exp, a_imp, b_imp, c_imp)

Generates an `IMEXARKTableau` struct from an IMEX ARK Butcher tableau. Only
`a_exp` and `a_imp` are required arguments; the default values for `b_exp` and
`b_imp` assume that the algorithm is FSAL (first same as last), and the default
values for `c_exp` and `c_imp` assume that the algorithm is internally
consistent.
"""
struct IMEXARKTableau{VS <: StaticArrays.StaticArray, MS <: StaticArrays.StaticArray} <: AbstractIMEXARKTableau
    a_exp::MS # matrix of size s×s
    b_exp::VS # vector of length s
    c_exp::VS # vector of length s
    a_imp::MS # matrix of size s×s
    b_imp::VS # vector of length s
    c_imp::VS # vector of length s
end
function IMEXARKTableau(;
    a_exp,
    b_exp = a_exp[end, :],
    c_exp = vec(sum(a_exp; dims = 2)),
    a_imp,
    b_imp = a_imp[end, :],
    c_imp = vec(sum(a_imp; dims = 2)),
)
    # TODO: add generic promote_eltype
    a_exp, a_imp = promote(a_exp, a_imp)
    b_exp, b_imp, c_exp, c_imp = promote(b_exp, b_imp, c_exp, c_imp)
    return IMEXARKTableau(a_exp, b_exp, c_exp, a_imp, b_imp, c_imp)
end

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
struct ARS111 <: AbstractIMEXARKTableau end

function tableau(::ARS111)
    IMEXARKTableau(; a_exp = @SArray([0 0; 1 0]), a_imp = @SArray([0 0; 0 1]))
end

"""
    ARS121

The Forward-Backward (1,2,1) implicit-explicit (IMEX) Runge-Kutta scheme of
[ARS1997](@cite), section 2.2.

This is equivalent to the `OrdinaryDiffEq.IMEXEulerARK` algorithm.
"""
struct ARS121 <: AbstractIMEXARKTableau end

function tableau(::ARS121)
    IMEXARKTableau(; a_exp = @SArray([0 0; 1 0]), b_exp = @SArray([0, 1]), a_imp = @SArray([0 0; 0 1]))
end

struct ARS122 <: AbstractIMEXARKTableau end
function tableau(::ARS122)
    IMEXARKTableau(;
        a_exp = @SArray([0 0; 1/2 0]),
        b_exp = @SArray([0, 1]),
        a_imp = @SArray([0 0; 0 1/2]),
        b_imp = @SArray([0, 1]),
    )
end

struct ARS233 <: AbstractIMEXARKTableau end
function tableau(::ARS233)
    γ = 1 / 2 + √3 / 6
    IMEXARKTableau(;
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
struct ARS232 <: AbstractIMEXARKTableau end
function tableau(::ARS232)
    γ = 1 - √2 / 2
    δ = -2√2 / 3
    IMEXARKTableau(;
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

struct ARS222 <: AbstractIMEXARKTableau end
function tableau(::ARS222)
    γ = 1 - √2 / 2
    δ = 1 - 1 / 2γ
    IMEXARKTableau(; a_exp = @SArray([
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
struct ARS343 <: AbstractIMEXARKTableau end
function tableau(::ARS343)
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
    return IMEXARKTableau(;
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

struct ARS443 <: AbstractIMEXARKTableau end
function tableau(::ARS443)
    IMEXARKTableau(;
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
function make_IMKGTableau(α, α̂, δ̂, β = ntuple(_ -> 0, length(δ̂)))
    s = length(α̂) + 1
    type = SMatrix{s, s}
    return IMEXARKTableau(;
        a_exp = sacollect(type, imkg_exp(i, j, α, β) for i in 1:s, j in 1:s),
        a_imp = sacollect(type, imkg_imp(i, j, α̂, β, δ̂) for i in 1:s, j in 1:s),
    )
end

struct IMKG232a <: AbstractIMEXARKTableau end
function tableau(::IMKG232a)
    make_IMKGTableau((1 / 2, 1 / 2, 1), (0, -1 / 2 + √2 / 2, 1), (1 - √2 / 2, 1 - √2 / 2))
end

struct IMKG232b <: AbstractIMEXARKTableau end
function tableau(::IMKG232b)
    make_IMKGTableau((1 / 2, 1 / 2, 1), (0, -1 / 2 - √2 / 2, 1), (1 + √2 / 2, 1 + √2 / 2))
end

struct IMKG242a <: AbstractIMEXARKTableau end
function tableau(::IMKG242a)
    make_IMKGTableau((1 / 4, 1 / 3, 1 / 2, 1), (0, 0, -1 / 2 + √2 / 2, 1), (0, 1 - √2 / 2, 1 - √2 / 2))
end

struct IMKG242b <: AbstractIMEXARKTableau end
function tableau(::IMKG242b)
    make_IMKGTableau((1 / 4, 1 / 3, 1 / 2, 1), (0, 0, -1 / 2 - √2 / 2, 1), (0, 1 + √2 / 2, 1 + √2 / 2))
end

# The paper uses √3/6 for α̂[3], which also seems to work.
struct IMKG243a <: AbstractIMEXARKTableau end
function tableau(::IMKG243a)
    make_IMKGTableau((1 / 4, 1 / 3, 1 / 2, 1), (0, 1 / 6, -√3 / 6, 1), (1 / 2 + √3 / 6, 1 / 2 + √3 / 6, 1 / 2 + √3 / 6))
end

struct IMKG252a <: AbstractIMEXARKTableau end
function tableau(::IMKG252a)
    make_IMKGTableau((1 / 4, 1 / 6, 3 / 8, 1 / 2, 1), (0, 0, 0, -1 / 2 + √2 / 2, 1), (0, 0, 1 - √2 / 2, 1 - √2 / 2))
end

struct IMKG252b <: AbstractIMEXARKTableau end
function tableau(::IMKG252b)
    make_IMKGTableau((1 / 4, 1 / 6, 3 / 8, 1 / 2, 1), (0, 0, 0, -1 / 2 - √2 / 2, 1), (0, 0, 1 + √2 / 2, 1 + √2 / 2))
end

# The paper uses 0.08931639747704086 for α̂[3], which also seems to work.
struct IMKG253a <: AbstractIMEXARKTableau end
function tableau(::IMKG253a)
    make_IMKGTableau(
        (1 / 4, 1 / 6, 3 / 8, 1 / 2, 1),
        (0, 0, √3 / 4 * (1 - √3 / 3) * ((1 + √3 / 3)^2 - 2), √3 / 6, 1),
        (0, 1 / 2 - √3 / 6, 1 / 2 - √3 / 6, 1 / 2 - √3 / 6),
    )
end

# The paper uses 1.2440169358562922 for α̂[3], which also seems to work.
struct IMKG253b <: AbstractIMEXARKTableau end
function tableau(::IMKG253b)
    make_IMKGTableau(
        (1 / 4, 1 / 6, 3 / 8, 1 / 2, 1),
        (0, 0, √3 / 4 * (1 + √3 / 3) * ((1 - √3 / 3)^2 - 2), -√3 / 6, 1),
        (0, 1 / 2 + √3 / 6, 1 / 2 + √3 / 6, 1 / 2 + √3 / 6),
    )
end

struct IMKG254a <: AbstractIMEXARKTableau end
function tableau(::IMKG254a)
    make_IMKGTableau((1 / 4, 1 / 6, 3 / 8, 1 / 2, 1), (0, -3 / 10, 5 / 6, -3 / 2, 1), (-1 / 2, 1, 1, 2))
end

struct IMKG254b <: AbstractIMEXARKTableau end
function tableau(::IMKG254b)
    make_IMKGTableau((1 / 4, 1 / 6, 3 / 8, 1 / 2, 1), (0, -1 / 20, 5 / 4, -1 / 2, 1), (-1 / 2, 1, 1, 1))
end

struct IMKG254c <: AbstractIMEXARKTableau end
function tableau(::IMKG254c)
    make_IMKGTableau((1 / 4, 1 / 6, 3 / 8, 1 / 2, 1), (0, 1 / 20, 5 / 36, 1 / 3, 1), (1 / 6, 1 / 6, 1 / 6, 1 / 6))
end

# The paper and HOMME completely disagree on this algorithm. Since the version
# in the paper is not "342" (it appears to be "332"), the version from HOMME is
# used here.
# const IMKG342a = make_IMKGTableau(
#     (0, 1/3, 1/3, 3/4),
#     (0, -1/6 - √3/6, -1/6 - √3/6, 3/4),
#     (0, 1/2 + √3/6, 1/2 + √3/6),
#     (1/3, 1/3, 1/4),
# )
struct IMKG342a <: AbstractIMEXARKTableau end
function tableau(::IMKG342a)
    make_IMKGTableau(
        (1 / 4, 2 / 3, 1 / 3, 3 / 4),
        (0, 1 / 6 - √3 / 6, -1 / 6 - √3 / 6, 3 / 4),
        (0, 1 / 2 + √3 / 6, 1 / 2 + √3 / 6),
        (0, 1 / 3, 1 / 4),
    )
end

struct IMKG343a <: AbstractIMEXARKTableau end
function tableau(::IMKG343a)
    make_IMKGTableau((1 / 4, 2 / 3, 1 / 3, 3 / 4), (0, -1 / 3, -2 / 3, 3 / 4), (-1 / 3, 1, 1), (0, 1 / 3, 1 / 4))
end

# The paper and HOMME completely disagree on this algorithm, but neither version
# is "353" (they appear to be "343" and "354", respectively).
# struct IMKG353a <: AbstractIMEXARKTableau end
# function tableau(::IMKG353a)
#     make_IMKGTableau(
#         (1/4, 2/3, 1/3, 3/4),
#         (0, -359/600, -559/600, 3/4),
#         (-1.1678009811335388, 253/200, 253/200),
#         (0, 1/3, 1/4),
#     )
# end
# struct IMKG353a <: AbstractIMEXARKTableau end
# function tableau(::IMKG353a)
#     make_IMKGTableau(
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
# struct IMKG354a <: AbstractIMEXARKTableau end
# function tableau(::IMKG354a)
#     make_IMKGTableau(
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

struct DBM453 <: AbstractIMEXARKTableau end
function tableau(::DBM453)
    γ = 0.32591194130117247
    IMEXARKTableau(;
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

struct HOMMEM1 <: AbstractIMEXARKTableau end
function tableau(::HOMMEM1)
    IMEXARKTableau(;
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
