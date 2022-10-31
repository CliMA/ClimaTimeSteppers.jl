export UpdateSignal, NewStep, NewNewtonSolve, NewNewtonIteration
export UpdateSignalHandler, UpdateEvery, UpdateEveryN

"""
    UpdateSignal

A signal that gets passed to an `UpdateSignalHandler` whenever a certain
operation is performed.
"""
abstract type UpdateSignal end

"""
    NewStep()

The signal for a new time step.
"""
struct NewStep <: UpdateSignal end

"""
    NewNewtonSolve()

The signal for a new `run!` of Newton's method, which occurs on every implicit
Runge-Kutta stage of the integrator.
"""
struct NewNewtonSolve <: UpdateSignal end

"""
    NewNewtonIteration()

The signal for a new iteration of Newton's method.
"""
struct NewNewtonIteration <: UpdateSignal end

"""
    UpdateSignalHandler

Updates a value upon receiving an appropriate `UpdateSignal`. This is done by
calling `run!(::UpdateSignalHandler, cache, ::UpdateSignal, f!, args...)`, where
`f!` is function such that `f!(args...)` modifies the desired value in-place.
The `cache` can be obtained with `allocate_cache(::UpdateSignalHandler)`.
"""
abstract type UpdateSignalHandler end

"""
    UpdateEvery(update_signal)

An `UpdateSignalHandler` that executes the update every time it is `run!` with
`update_signal`.
"""
struct UpdateEvery{U <: UpdateSignal} <: UpdateSignalHandler
    update_signal::U
end

allocate_cache(::UpdateSignalHandler) = (;)

function run!(alg::UpdateEvery{U}, cache, ::U, f!, args...) where {
    U <: UpdateSignal,
}
    f!(args...)
    return true
end

"""
    UpdateEveryN(update_signal, n, reset_n_signal = nothing)

An `UpdateSignalHandler` that executes the update every `n`-th time it is `run!`
with `update_signal`. If `reset_n_signal` is specified, then the value of `n` is
reset to 0 every time the signal handler is `run!` with `reset_n_signal`.
"""
struct UpdateEveryN{U <: UpdateSignal, R <: Union{Nothing, UpdateSignal}} <:
    UpdateSignalHandler
    update_signal::U
    n::Int
    reset_n_signal::R
end
UpdateEveryN(update_signal, n, reset_n_signal = nothing) =
    UpdateEveryN(update_signal, n, reset_n_signal)

allocate_cache(::UpdateEveryN) = (; n = Ref(0))

function run!(alg::UpdateEveryN{U}, cache, ::U, f!, args...) where {
    U <: UpdateSignal,
}
    cache.n[] += 1
    if cache.n[] == alg.n
        f!(args...)
        cache.n[] = 0
        return true
    end
    return false
end
function run!(alg::UpdateEveryN{U, R}, cache, ::R, f!, args...) where {
    U,
    R <: UpdateSignal,
}
    cache.n[] = 0
    return false
end
