export UpdateSignal, NewTimeStep, NewNewtonSolve, NewNewtonIteration
export UpdateSignalHandler, UpdateEvery, UpdateEveryN, UpdateEveryDt

"""
    UpdateSignal

A signal that gets passed to an `UpdateSignalHandler` whenever a certain
operation is performed.
"""
abstract type UpdateSignal end

"""
    NewTimeStep(t)

The signal for a new time step at time `t`.
"""
struct NewTimeStep{T} <: UpdateSignal
    t::T
end

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
The `cache` can be obtained with `allocate_cache(::UpdateSignalHandler, FT)`,
where `FT` is the floating-point type of the integrator.
"""
abstract type UpdateSignalHandler end

"""
    UpdateEvery(update_signal_type)

An `UpdateSignalHandler` that performs the update whenever it is `run!` with an
`UpdateSignal` of type `update_signal_type`.
"""
struct UpdateEvery{U <: UpdateSignal} <: UpdateSignalHandler end
UpdateEvery(::Type{U}) where {U} = UpdateEvery{U}()

run!(alg::UpdateEvery{U}, cache, ::U, f!, args...) where {U} = f!(args...)

"""
    UpdateEveryN(n, update_signal_type, reset_signal_type = Nothing)

An `UpdateSignalHandler` that performs the update every `n`-th time it is `run!`
with an `UpdateSignal` of type `update_signal_type`. If `reset_signal_type` is
specified, then the counter (which gets incremented from 0 to `n` and then gets
reset to 0 when it is time to perform another update) is reset to 0 whenever the
signal handler is `run!` with an `UpdateSignal` of type `reset_signal_type`.
"""
struct UpdateEveryN{U <: UpdateSignal, R <: Union{Nothing, UpdateSignal}} <: UpdateSignalHandler
    n::Int
end
UpdateEveryN(n, ::Type{U}, ::Type{R} = Nothing) where {U, R} = UpdateEveryN{U, R}(n)

allocate_cache(::UpdateEveryN, _) = (; counter = Ref(0))

function run!(alg::UpdateEveryN{U}, cache, ::U, f!, args...) where {U}
    (; n) = alg
    (; counter) = cache
    if counter[] == 0
        f!(args...)
    end
    counter[] += 1
    if counter[] == n
        counter[] = 0
    end
end
function run!(alg::UpdateEveryN{<:Any, R}, cache, ::R, f!, args...) where {R}
    (; counter) = cache
    counter[] = 0
end

"""
    UpdateEveryDt(dt)

An `UpdateSignalHandler` that performs the update whenever it is `run!` with an
`UpdateSignal` of type `NewTimeStep` and the difference between the current time
and the previous update time is no less than `dt`.
"""
struct UpdateEveryDt{T} <: UpdateSignalHandler
    dt::T
end

# TODO: This assumes that typeof(t) == FT, which might not always be correct.
allocate_cache(alg::UpdateEveryDt, ::Type{FT}) where {FT} = (; is_first_t = Ref(true), prev_update_t = Ref{FT}())

function run!(alg::UpdateEveryDt, cache, signal::NewTimeStep, f!, args...)
    (; dt) = alg
    (; is_first_t, prev_update_t) = cache
    (; t) = signal
    if is_first_t[] || abs(t - prev_update_t[]) >= dt
        f!(args...)
        is_first_t[] = false
        prev_update_t[] = t
    end
end
