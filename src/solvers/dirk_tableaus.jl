export DIRKTableau, DIRKAlgorithmName
export ESDIRK43

"""
    DIRKTableau(; a, b, c)

Butcher tableau for a (diagonally) implicit Runge-Kutta method:

    Yᵢ = uₙ + dt * sumⱼ a[i, j] * F(Yⱼ, tₙ + dt*c[j])
    uₙ₊₁ = uₙ + dt * sumᵢ b[i] * F(Yᵢ, tₙ + dt*c[i])

The matrix ``a`` must be lower triangular (DIRK/ESDIRK).
"""
struct DIRKTableau{AT <: SPCO, BT <: SPCO, CT <: SPCO}
    a::AT # s×s lower triangular
    b::BT # length s
    c::CT # length s
end

DIRKTableau(args...) = DIRKTableau(map(x -> SparseCoeffs(x), args)...)

function DIRKTableau(; a, b = a[end, :], c = vec(sum(a; dims = 2)))
    @assert all(iszero, UpperTriangular(a))
    return DIRKTableau(a, b, c)
end

abstract type DIRKAlgorithmName <: AbstractAlgorithmName end

################################################################################
# ESDIRK methods
################################################################################

"""
    ESDIRK43

ESDIRK4(3)6L[2]SA (explicit first stage, L-stable, stiffly accurate).

Advancing method order: 4
Embedded method order: 3 (error estimate not implemented in this repository)
"""
struct ESDIRK43 <: DIRKAlgorithmName end

function DIRKTableau(::ESDIRK43)
    # Coefficients taken from PathSim's ESDIRK43 implementation, attributed to
    # Carpenter & Kennedy (2019).
    s = 6
    a = zeros(Float64, s, s)

    γ = 0.25
    for i in 2:s
        a[i, i] = γ
    end

    # Stage 2
    a[2, 1] = 0.25

    # Stage 3
    a21 = -1356991263433 / 26208533697614
    a[3, 1] = a21
    a[3, 2] = a21

    # Stage 4
    a31 = -1778551891173 / 14697912885533
    a[4, 1] = a31
    a[4, 2] = a31
    a[4, 3] = 7325038566068 / 12797657924939

    # Stage 5
    a41 = -24076725932807 / 39344244018142
    a[5, 1] = a41
    a[5, 2] = a41
    a[5, 3] = 9344023789330 / 6876721947151
    a[5, 4] = 11302510524611 / 18374767399840

    # Stage 6 (stiffly accurate: b = last row)
    a51 = 657241292721 / 9909463049845
    a[6, 1] = a51
    a[6, 2] = a51
    a[6, 3] = 1290772910128 / 5804808736437
    a[6, 4] = 1103522341516 / 2197678446715
    a[6, 5] = -3 / 28

    return DIRKTableau(; a)
end

tableau(name::DIRKAlgorithmName) = DIRKTableau(name)

