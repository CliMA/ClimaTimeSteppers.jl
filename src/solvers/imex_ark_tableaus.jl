export ARS111, ARS121, ARS122, ARS233, ARS232, ARS222, ARS343, ARS443
export IMKG232a, IMKG232b, IMKG242a, IMKG242b, IMKG252a, IMKG252b
export IMKG253a, IMKG253b, IMKG254a, IMKG254b, IMKG254c, IMKG342a, IMKG343a
export DBM453

using StaticArrays: @SArray, SMatrix, sacollect

################################################################################

# ARS algorithms

# From Section 2 of "Implicit-Explicit Runge-Kutta Methods for Time-Dependent
# Partial Differential Equations" by Ascher et al.

# Each algorithm is named ARSsσp, where s is the number of implicit stages, σ is
# the number of explicit stages, and p is the order of accuracy

# This algorithm is equivalent to OrdinaryDiffEq.IMEXEuler.
const ARS111 = make_IMEXARKAlgorithm(;
    a_exp = @SArray([0 0; 1 0]),
    a_imp = @SArray([0 0; 0 1]),
)

const ARS121 = make_IMEXARKAlgorithm(;
    a_exp = @SArray([0 0; 1 0]),
    b_exp = @SArray([0, 1]),
    a_imp = @SArray([0 0; 0 1]),
)

const ARS122 = make_IMEXARKAlgorithm(;
    a_exp = @SArray([0 0; 1/2 0]),
    b_exp = @SArray([0, 1]),
    a_imp = @SArray([0 0; 0 1/2]),
    b_imp = @SArray([0, 1]),
)

const ARS233 = let
    γ = 1/2 + √3/6
    make_IMEXARKAlgorithm(;
        a_exp = @SArray([
            0     0      0;
            γ     0      0;
            (γ-1) (2-2γ) 0;
        ]),
        b_exp = @SArray([0, 1/2, 1/2]),
        a_imp = @SArray([
            0 0      0;
            0 γ      0;
            0 (1-2γ) γ;
        ]),
        b_imp = @SArray([0, 1/2, 1/2]),
    )
end

const ARS232 = let
    γ = 1 - √2/2
    δ = -2√2/3
    make_IMEXARKAlgorithm(;
        a_exp = @SArray([
            0 0     0;
            γ 0     0;
            δ (1-δ) 0;
        ]),
        b_exp = @SArray([0, 1 - γ, γ]),
        a_imp = @SArray([
            0 0     0;
            0 γ     0;
            0 (1-γ) γ;
        ]),
    )
end

const ARS222 = let
    γ = 1 - √2/2
    δ = 1 - 1/2γ
    make_IMEXARKAlgorithm(;
        a_exp = @SArray([
            0 0     0;
            γ 0     0;
            δ (1-δ) 0;
        ]),
        a_imp = @SArray([
            0 0     0;
            0 γ     0;
            0 (1-γ) γ;
        ]),
    )
end

const ARS343 = let
    γ = 0.4358665215084590
    a42 = 0.5529291480359398
    a43 = 0.5529291480359398
    b1 = -3/2 * γ^2 + 4 * γ - 1/4
    b2 =  3/2 * γ^2 - 5 * γ + 5/4
    a31 = (1 - 9/2 * γ + 3/2 * γ^2) * a42 +
        (11/4 - 21/2 * γ + 15/4 * γ^2) * a43 - 7/2 + 13 * γ - 9/2 * γ^2
    a32 = (-1 + 9/2 * γ - 3/2 * γ^2) * a42 +
        (-11/4 + 21/2 * γ - 15/4 * γ^2) * a43 + 4 - 25/2 * γ + 9/2 * γ^2
    a41 = 1 - a42 - a43
    make_IMEXARKAlgorithm(;
        a_exp = @SArray([
            0   0   0   0;
            γ   0   0   0;
            a31 a32 0   0;
            a41 a42 a43 0;
        ]),
        b_exp = @SArray([0, b1, b2, γ]),
        a_imp = @SArray([
            0 0       0  0;
            0 γ       0  0;
            0 (1-γ)/2 γ  0;
            0 b1      b2 γ;
        ]),
    )
end

const ARS443 = make_IMEXARKAlgorithm(;
    a_exp = @SArray([
        0     0    0   0    0;
        1/2   0    0   0    0;
        11/18 1/18 0   0    0;
        5/6   -5/6 1/2 0    0;
        1/4   7/4  3/4 -7/4 0;
    ]),
    a_imp = @SArray([
        0 0    0    0   0;
        0 1/2  0    0   0;
        0 1/6  1/2  0   0;
        0 -1/2 1/2  1/2 0;
        0 3/2  -3/2 1/2 1/2;
    ]),
)

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

imkg_exp(i, j, α, β) = i == j + 1 ? α[j] : (i > 2 && j == 1 ? β[i - 2] : 0)
imkg_imp(i, j, α̂, β, δ̂) = i == j + 1 ? α̂[j] :
    (i > 2 && j == 1 ? β[i - 2] : (1 < i <= length(α̂) && i == j ? δ̂[i - 1] : 0))
function make_IMKGAlgorithm(α, α̂, δ̂, β = ntuple(_ -> 0, length(δ̂)))
    s = length(α̂) + 1
    type = SMatrix{s, s}
    return make_IMEXARKAlgorithm(;
        a_exp = sacollect(type, imkg_exp(i, j, α, β) for i in 1:s, j in 1:s),
        a_imp = sacollect(type, imkg_imp(i, j, α̂, β, δ̂) for i in 1:s, j in 1:s),
    )
end

const IMKG232a = make_IMKGAlgorithm(
    (1/2, 1/2, 1),
    (0, -1/2 + √2/2, 1),
    (1 - √2/2, 1 - √2/2),
)

const IMKG232b = make_IMKGAlgorithm(
    (1/2, 1/2, 1),
    (0, -1/2 - √2/2, 1),
    (1 + √2/2, 1 + √2/2),
)

const IMKG242a = make_IMKGAlgorithm(
    (1/4, 1/3, 1/2, 1),
    (0, 0, -1/2 + √2/2, 1),
    (0, 1 - √2/2, 1 - √2/2),
)

const IMKG242b = make_IMKGAlgorithm(
    (1/4, 1/3, 1/2, 1),
    (0, 0, -1/2 - √2/2, 1),
    (0, 1 + √2/2, 1 + √2/2),
)

# The paper uses √3/6 for α̂[3], which also seems to work.
const IMKG243a = make_IMKGAlgorithm(
    (1/4, 1/3, 1/2, 1),
    (0, 1/6, -√3/6, 1),
    (1/2 + √3/6, 1/2 + √3/6, 1/2 + √3/6),
)

const IMKG252a = make_IMKGAlgorithm(
    (1/4, 1/6, 3/8, 1/2, 1),
    (0, 0, 0, -1/2 + √2/2, 1),
    (0, 0, 1 - √2/2, 1 - √2/2),
)

const IMKG252b = make_IMKGAlgorithm(
    (1/4, 1/6, 3/8, 1/2, 1),
    (0, 0, 0, -1/2 - √2/2, 1),
    (0, 0, 1 + √2/2, 1 + √2/2),
)

# The paper uses 0.08931639747704086 for α̂[3], which also seems to work.
const IMKG253a = make_IMKGAlgorithm(
    (1/4, 1/6, 3/8, 1/2, 1),
    (0, 0, √3/4 * (1 - √3/3) * ((1 + √3/3)^2 - 2), √3/6, 1),
    (0, 1/2 - √3/6, 1/2 - √3/6, 1/2 - √3/6),
)

# The paper uses 1.2440169358562922 for α̂[3], which also seems to work.
const IMKG253b = make_IMKGAlgorithm(
    (1/4, 1/6, 3/8, 1/2, 1),
    (0, 0, √3/4 * (1 + √3/3) * ((1 - √3/3)^2 - 2), -√3/6, 1),
    (0, 1/2 + √3/6, 1/2 + √3/6, 1/2 + √3/6),
)

const IMKG254a = make_IMKGAlgorithm(
    (1/4, 1/6, 3/8, 1/2, 1),
    (0, -3/10, 5/6, -3/2, 1),
    (-1/2, 1, 1, 2),
)

const IMKG254b = make_IMKGAlgorithm(
    (1/4, 1/6, 3/8, 1/2, 1),
    (0, -1/20, 5/4, -1/2, 1),
    (-1/2, 1, 1, 1),
)

const IMKG254c = make_IMKGAlgorithm(
    (1/4, 1/6, 3/8, 1/2, 1),
    (0, 1/20, 5/36, 1/3, 1),
    (1/6, 1/6, 1/6, 1/6),
)

# The paper and HOMME completely disagree on this algorithm. Since the version
# in the paper is not "342" (it appears to be "332"), the version from HOMME is
# used here.
# const IMKG342a = make_IMKGAlgorithm(
#     (0, 1/3, 1/3, 3/4),
#     (0, -1/6 - √3/6, -1/6 - √3/6, 3/4),
#     (0, 1/2 + √3/6, 1/2 + √3/6),
#     (1/3, 1/3, 1/4),
# )
const IMKG342a = make_IMKGAlgorithm(
    (1/4, 2/3, 1/3, 3/4),
    (0, 1/6 - √3/6, -1/6 - √3/6, 3/4),
    (0, 1/2 + √3/6, 1/2 + √3/6),
    (0, 1/3, 1/4),
)

const IMKG343a = make_IMKGAlgorithm(
    (1/4, 2/3, 1/3, 3/4),
    (0, -1/3, -2/3, 3/4),
    (-1/3, 1, 1),
    (0, 1/3, 1/4),
)

# The paper and HOMME completely disagree on this algorithm, but neither version
# is "353" (they appear to be "343" and "354", respectively).
# const IMKG353a = make_IMKGAlgorithm(
#     (1/4, 2/3, 1/3, 3/4),
#     (0, -359/600, -559/600, 3/4),
#     (-1.1678009811335388, 253/200, 253/200),
#     (0, 1/3, 1/4),
# )
# const IMKG353a = make_IMKGAlgorithm(
#     (-0.017391304347826087, -23/25, 5/3, 1/3, 3/4),
#     (0.3075640504095504, -1.2990164859879263, 751/600, -49/60, 3/4),
#     (-0.2981612530370581, 83/200, 83/200, 23/20),
#     (1, -1, 1/3, 1/4),
# )

# The version of this algorithm in the paper is not "354" (it appears to be
# "253"), and this algorithm is missing from HOMME (or, more precisely, the
# tableau for IMKG353a is mistakenly used to define IMKG354a, and the tableau
# for IMKG354a is not specified).
# const IMKG354a = make_IMKGAlgorithm(
#     (1/5, 1/5, 2/3, 1/3, 3/4),
#     (0, 0, 11/30, -2/3, 3/4),
#     (0, 2/4, 2/5, 1),
#     (0, 0, 1/3, 1/4),
# )

################################################################################

# DBM algorithm

# From Appendix A of "Evaluation of Implicit-Explicit Additive Runge-Kutta
# Integrators for the HOMME-NH Dynamical Core" by Vogl et al.

# The algorithm has 4 implicit stages, 5 overall stages, and 3rd order accuracy.

const DBM453 = let
    γ = 0.32591194130117247
    make_IMEXARKAlgorithm(;
        a_exp = @SArray([
            0                    0                   0                   0                    0;
            0.10306208811591838  0                   0                   0                    0;
            -0.94124866143519894 1.6626399742527356  0                   0                    0;
            -1.3670975201437765  1.3815852911016873  1.2673234025619065  0                    0;
            -0.81287582068772448 0.81223739060505738 0.90644429603699305 0.094194134045674111 0;
        ]),
        b_exp = @SArray([
            0.87795339639076675, -0.72692641526151547, 0.7520413715737272, -0.22898029400415088, γ
        ]),
        a_imp = @SArray([
            0                    0                    0                   0                    0;
            -0.2228498531852541  γ                    0                   0                    0;
            -0.46801347074080545 0.86349284225716961  γ                   0                    0;
            -0.46509906651927421 0.81063103116959553  0.61036726756832357 γ                    0;
            0.87795339639076675  -0.72692641526151547 0.7520413715737272  -0.22898029400415088 γ;
        ]),
    )
end
