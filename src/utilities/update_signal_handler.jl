export UpdateSignal, NewTimeStep, NewNewtonSolve, NewNewtonIteration
export EndOfStage, EndOfStep, WithDSS
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

# Signal hierarchy for DSS-related events:
#
#   EndOfStep <: EndOfStage <: WithDSS <: UpdateSignal
#
# Because `UpdateEvery`'s dispatch responds to any *subtype* of its target,
# a broader handler automatically fires on narrower events. For example
# `UpdateEvery(WithDSS)` fires at every DSS site (including `EndOfStage`
# and `EndOfStep`), while `UpdateEvery(EndOfStep)` fires only at end of step.
# Concrete singleton subtypes (`WithDSSSignal`, `EndOfStageSignal`,
# `EndOfStepSignal`) are what the integrator actually fires at call sites.

"""
    WithDSS

Broadest DSS-site signal type. Use `UpdateEvery(WithDSS)` to fire a hook
at every `dss!` call (including pre-implicit and post-`initialize_imp!`).
"""
abstract type WithDSS <: UpdateSignal end

"""
    EndOfStage

Signal type for state-ready-for-tendency-evaluation moments (post-Newton
in implicit stages, right after DSS in explicit-only stages, and end of
step). A subset of [`WithDSS`](@ref).
"""
abstract type EndOfStage <: WithDSS end

"""
    EndOfStep

Signal type for the end of a full time step. A subset of [`EndOfStage`](@ref).
"""
abstract type EndOfStep <: EndOfStage end

# Concrete singleton signal instances (fired internally at call sites).
struct WithDSSSignal <: WithDSS end
struct EndOfStageSignal <: EndOfStage end
struct EndOfStepSignal <: EndOfStep end

"""
    UpdateEvery(update_signal_type)

Perform the update whenever `needs_update!` receives an `UpdateSignal`
whose type is `update_signal_type` **or a subtype of it**. The subtype
matching means a broad handler like `UpdateEvery(WithDSS)` fires on any
`EndOfStage` / `EndOfStep` event as well.
"""
struct UpdateEvery{U <: UpdateSignal} <: UpdateSignalHandler end
UpdateEvery(::Type{U}) where {U} = UpdateEvery{U}()

needs_update!(alg::UpdateEvery{U}, ::S) where {U <: UpdateSignal, S <: U} = true

"""
    UpdateEveryN(n, update_signal_type, reset_signal_type = Nothing)

Perform the update every `n`-th time `needs_update!` receives an
`UpdateSignal` of type `update_signal_type`. The internal counter cycles
through `0, 1, …, n-1`, and the update is performed when the counter is `0`.
If `reset_signal_type` is specified, the counter is also reset to `0` whenever
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
function needs_update!(alg::UpdateEveryN{U, C, R}, ::R) where {U, C, R <: UpdateSignal}
    (; counter) = alg
    counter[] = 0
    return false
end

# Account for method ambiguity:
needs_update!(::UpdateEveryN{U, C, U}, ::U) where {U <: UpdateSignal, C} =
    error("Reset and update signal types cannot be the same.")

"""
    UpdateEveryDt(dt)

Perform the update whenever `needs_update!` receives a [`NewTimeStep`](@ref)
signal and the elapsed time since the last update is at least `dt`. The first
`NewTimeStep` always performs the update.
"""
struct UpdateEveryDt{T, BR, FTR} <: UpdateSignalHandler
    dt::T
    is_first_t::BR
    prev_update_t::FTR
end
UpdateEveryDt(dt::Real) = UpdateEveryDt(dt, Ref(true), Ref{typeof(dt)}())

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
