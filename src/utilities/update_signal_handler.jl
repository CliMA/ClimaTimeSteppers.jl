export UpdateSignal, NewTimeStep, NewNewtonSolve, NewNewtonIteration
export UpdateSignalHandler, UpdateEvery, UpdateEveryN, UpdateEveryDt

"""
    UpdateSignal

A signal that gets passed to an `UpdateSignalHandler` whenever a certain
operation is performed.
"""
abstract type UpdateSignal end

"""
    UpdateSignalHandler

A boolean indicating if updates a value upon receiving an appropriate
`UpdateSignal`. This is done by calling
`needs_update!(::UpdateSignalHandler, cache, ::UpdateSignal)`.

The `cache` can be obtained with `allocate_cache(::UpdateSignalHandler, FT)`,
where `FT` is the floating-point type of the integrator.
"""
abstract type UpdateSignalHandler end

needs_update!(::UpdateSignalHandler, cache, ::UpdateSignal) = false

"""
    NewTimeStep(t)

The signal for a new time step at time `t`.
"""
struct NewTimeStep{T} <: UpdateSignal
    t::T
end

"""
    NewNewtonSolve()

The signal for a new `needs_update!` of Newton's method, which occurs on every implicit
Runge-Kutta stage of the integrator.
"""
struct NewNewtonSolve <: UpdateSignal end

"""
    NewNewtonIteration()

The signal for a new iteration of Newton's method.
"""
struct NewNewtonIteration <: UpdateSignal end

"""
    UpdateEvery(update_signal_type)

An `UpdateSignalHandler` that performs the update whenever it is `needs_update!` with an
`UpdateSignal` of type `update_signal_type`.
"""
struct UpdateEvery{U <: UpdateSignal} <: UpdateSignalHandler end
UpdateEvery(::Type{U}) where {U} = UpdateEvery{U}()

allocate_cache(::UpdateEvery, _) = nothing

needs_update!(alg::UpdateEvery{U}, cache, ::U) where {U <: UpdateSignal} = true

"""
    UpdateEveryN(n, update_signal_type, reset_signal_type = Nothing)

An `UpdateSignalHandler` that performs the update every `n`-th time it is `needs_update!`
with an `UpdateSignal` of type `update_signal_type`. If `reset_signal_type` is
specified, then the counter (which gets incremented from 0 to `n` and then gets
reset to 0 when it is time to perform another update) is reset to 0 whenever the
signal handler is `needs_update!` with an `UpdateSignal` of type `reset_signal_type`.
"""
struct UpdateEveryN{U <: UpdateSignal, R <: Union{Nothing, UpdateSignal}} <: UpdateSignalHandler
    n::Int
end
UpdateEveryN(n, ::Type{U}, ::Type{R} = Nothing) where {U, R} = UpdateEveryN{U, R}(n)

allocate_cache(::UpdateEveryN, _) = (; counter = Ref(0))

function needs_update!(alg::UpdateEveryN{U}, cache, ::U) where {U <: UpdateSignal}
    (; n) = alg
    (; counter) = cache
    result = counter[] == 0
    counter[] += 1
    if counter[] == n
        counter[] = 0
    end
    return result
end
function needs_update!(alg::UpdateEveryN{U, R}, cache, ::R) where {U, R <: UpdateSignal}
    (; counter) = cache
    counter[] = 0
    return false
end

# Account for method ambiguitiy:
needs_update!(::UpdateEveryN{U, U}, cache, ::U) where {U <: UpdateSignal} =
    error("Reset and update signal types cannot be the same.")

"""
    UpdateEveryDt(dt)

An `UpdateSignalHandler` that performs the update whenever it is `needs_update!` with an
`UpdateSignal` of type `NewTimeStep` and the difference between the current time
and the previous update time is no less than `dt`.
"""
struct UpdateEveryDt{T} <: UpdateSignalHandler
    dt::T
end

# TODO: This assumes that typeof(t) == FT, which might not always be correct.
allocate_cache(alg::UpdateEveryDt, ::Type{FT}) where {FT} = (; is_first_t = Ref(true), prev_update_t = Ref{FT}())

function needs_update!(alg::UpdateEveryDt, cache, signal::NewTimeStep)
    (; dt) = alg
    (; is_first_t, prev_update_t) = cache
    (; t) = signal
    result = false
    if is_first_t[] || abs(t - prev_update_t[]) >= dt
        result = true
        is_first_t[] = false
        prev_update_t[] = t
    end
    return result
end
