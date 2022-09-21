export UpdateSignal, NewStep, NewStage, NewNewtonIteration
export UpdateSignalHandler, UpdateEvery, UpdateEveryN, can_handle

abstract type UpdateSignal end
struct NewStep <: UpdateSignal end
struct NewStage <: UpdateSignal end
struct NewNewtonIteration <: UpdateSignal end

abstract type UpdateSignalHandler end
struct UpdateEvery{U <: UpdateSignal} <: UpdateSignalHandler
    update_signal::U
end
struct UpdateEveryN{U <: UpdateSignal, R <: Union{Nothing, UpdateSignal}} <:
       UpdateSignalHandler
    update_signal::U
    n::Int
    reset_n_signal::R
end
UpdateEveryN(update_signal, n, reset_n_signal = nothing) =
    UpdateEveryN(update_signal, n, reset_n_signal)

can_handle(::Type{<:UpdateSignalHandler}, ::UpdateSignal) = false
can_handle(::Type{UpdateEvery{U}}, ::U) where {U <: UpdateSignal} = true
can_handle(::Type{UpdateEveryN{U}}, ::U) where {U <: UpdateSignal} = true

allocate_cache(::UpdateSignalHandler) = (;)
allocate_cache(::UpdateEveryN) = (; n = Ref(0))

run!(alg::UpdateSignalHandler, cache, ::UpdateSignal, f!, args...) = false
function run!(alg::UpdateEvery{U}, cache, ::U, f!, args...) where {U <: UpdateSignal}
    f!(args...)
    return true
end
function run!(alg::UpdateEveryN{U}, cache, ::U, f!, args...) where {U <: UpdateSignal}
    cache.n[] += 1
    if cache.n[] == alg.n
        f!(args...)
        cache.n[] = 0
        return true
    end
    return false
end
function run!(alg::UpdateEveryN{U, R}, cache, ::R, f!, args...) where {U, R <: UpdateSignal}
    cache.n[] = 0
    return false
end
