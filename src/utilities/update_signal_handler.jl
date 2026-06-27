export UpdateSignal, NewTimeStep, NewNewtonSolve, NewNewtonIteration
export UpdateSignalHandler, UpdateEvery, UpdateEveryN, UpdateEveryDt

"""
    UpdateSignal

A signal that gets passed to an `UpdateSignalHandler`
whenever a certain operation is performed.
"""
abstract type UpdateSignal end

"""
    UpdateSignalHandler

Decide whether to update a value upon receiving an appropriate
[`UpdateSignal`](@ref). Query via
`needs_update!(::UpdateSignalHandler, ::UpdateSignal)`.
"""
abstract type UpdateSignalHandler end

needs_update!(::UpdateSignalHandler, ::UpdateSignal) = false

"""
    NewTimeStep(t)

The signal for a new time step at time `t`.
"""
struct NewTimeStep{T} <: UpdateSignal
    t::T
end

"""
    NewNewtonSolve()

Signal emitted at the start of each Newton solve (once per implicit
Runge-Kutta stage).
"""
struct NewNewtonSolve <: UpdateSignal end

"""
    NewNewtonIteration()

The signal for a new iteration of Newton's method.
"""
struct NewNewtonIteration <: UpdateSignal end

"""
    UpdateEvery(update_signal_type)

Perform the update whenever `needs_update!` receives an `UpdateSignal` of
type `update_signal_type`.
"""
struct UpdateEvery{U <: UpdateSignal} <: UpdateSignalHandler end
UpdateEvery(::Type{U}) where {U} = UpdateEvery{U}()

needs_update!(alg::UpdateEvery{U}, ::U) where {U <: UpdateSignal} = true

"""
    UpdateEveryN(n, update_signal_type, reset_signal_type = Nothing)

Perform the update every `n`-th time `needs_update!` receives an
`UpdateSignal` of type `update_signal_type`. The internal counter increments
from 0 to `n` and resets to 0 when it is time to update. If
`reset_signal_type` is specified, the counter is also reset to 0 whenever
`needs_update!` receives a signal of that type.
"""
struct UpdateEveryN{U <: UpdateSignal, C, R <: Union{Nothing, UpdateSignal}} <:
       UpdateSignalHandler
    n::Int
    counter::C
end
UpdateEveryN(n, ::Type{U}, ::Type{R} = Nothing) where {U, R} =
    UpdateEveryN{U, typeof(Ref(0)), R}(n, Ref(0))

function needs_update!(alg::UpdateEveryN{U}, ::U) where {U <: UpdateSignal}
    (; n, counter) = alg
    result = counter[] == 0
    counter[] += 1
    if counter[] == n
        counter[] = 0
    end
    return result
end
function needs_update!(alg::UpdateEveryN{U, R}, ::R) where {U, R <: UpdateSignal}
    (; counter) = alg
    counter[] = 0
    return false
end

# Account for method ambiguity:
needs_update!(::UpdateEveryN{U, U}, ::U) where {U <: UpdateSignal} =
    error("Reset and update signal types cannot be the same.")

"""
    UpdateEveryDt(dt)

Perform the update whenever `needs_update!` receives a [`NewTimeStep`](@ref)
signal and the elapsed time since the last update is at least `dt`.

!!! note
    The constructor accepts a *type* argument: `UpdateEveryDt(dt::Type{FT})`.
    For example, `UpdateEveryDt(Float64)` creates a handler with an
    uninitialized `prev_update_t` reference.
"""
struct UpdateEveryDt{T, BR, FTR} <: UpdateSignalHandler
    dt::T
    is_first_t::BR
    prev_update_t::FTR
end
UpdateEveryDt(dt::Type{FT}) where {FT} = UpdateEveryDt(dt, Ref(true), Ref{FT}())

function needs_update!(alg::UpdateEveryDt, signal::NewTimeStep)
    (; dt, is_first_t, prev_update_t) = alg
    (; t) = signal
    result = false
    if is_first_t[] || abs(t - prev_update_t[]) >= dt
        result = true
        is_first_t[] = false
        prev_update_t[] = t
    end
    return result
end
