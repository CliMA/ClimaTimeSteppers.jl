struct IMEXTableaus{VS <: StaticArrays.StaticArray, MS <: StaticArrays.StaticArray}
    a_exp::MS # matrix of size s×s
    b_exp::VS # vector of length s
    c_exp::VS # vector of length s
    a_imp::MS # matrix of size s×s
    b_imp::VS # vector of length s
    c_imp::VS # vector of length s
end
function IMEXTableaus(;
    a_exp,
    b_exp = a_exp[end, :],
    c_exp = vec(sum(a_exp; dims = 2)),
    a_imp,
    b_imp = a_imp[end, :],
    c_imp = vec(sum(a_imp; dims = 2)),
)
    a_exp, a_imp = promote(a_exp, a_imp)
    b_exp, b_imp, c_exp, c_imp = promote(b_exp, b_imp, c_exp, c_imp)
    return IMEXTableaus(a_exp, b_exp, c_exp, a_imp, b_imp, c_imp)
end

abstract type AbstractAlgorithmName end
abstract type ARKAlgorithmName <: AbstractAlgorithmName end
abstract type SSPRKAlgorithmName <: AbstractAlgorithmName end
struct ARS222 <: ARKAlgorithmName end
function IMEXTableaus(::ARS222)
    γ = 1 - √2 / 2
    δ = 1 - 1 / 2γ
    IMEXTableaus(; a_exp = @SArray([
        0 0 0
        γ 0 0
        δ (1-δ) 0
    ]), a_imp = @SArray([
        0 0 0
        0 γ 0
        0 (1-γ) γ
    ]))
end
struct SSP222 <: SSPRKAlgorithmName end
function IMEXTableaus(::SSP222)
    γ = 1 - √2 / 2
    return IMEXTableaus(;
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

abstract type AbstractAlgorithmMode end
struct ARK <: AbstractAlgorithmMode end
struct SSPRK <: AbstractAlgorithmMode end
default_mode(::ARKAlgorithmName) = ARK()
default_mode(::SSPRKAlgorithmName) = SSPRK()

struct IMEXAlgorithm{M <: AbstractAlgorithmMode, T <: IMEXTableaus, N <: NewtonsMethod} <: DistributedODEAlgorithm
    mode::M
    tableaus::T
    newtons_method::N
end
IMEXAlgorithm(name::AbstractAlgorithmName, newtons_method::NewtonsMethod; mode = default_mode(name)) =
    IMEXAlgorithm(mode, IMEXTableaus(name), newtons_method)

function init_cache(prob::DiffEqBase.AbstractODEProblem, alg::IMEXAlgorithm{ARK}; kwargs...)
    # do stuff...
    return IMEXARKCache(U, T_lim, T_exp, T_imp, temp, γ, newtons_method_cache)
end

function init_cache(prob::DiffEqBase.AbstractODEProblem, alg::IMEXAlgorithm{SSPRK}; kwargs...)
    # do stuff...
    return IMEXSSPRKCache(U, U_exp, U_lim, T_lim, T_exp, T_imp, temp, γ, newtons_method_cache)
end
