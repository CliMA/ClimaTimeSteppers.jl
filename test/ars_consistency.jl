import ClimaTimeSteppers as CTS

"""
    CacheState
A state for the cache, to indicate when we
need to re-call `set_precomputed_quantities!`:
"""
mutable struct CacheState{FV, FT, PSCBC}
    u::FV
    stale::Bool
    called_in_IC::Bool
    pscb_called::PSCBC
    t_last_called::FT
end

function CacheState(u, pscb_called, t_start)
    fv = similar(u)
    @. fv = u
    return CacheState(fv, true, false, pscb_called, t_start)
end

alleqeq(a, b) = all(x -> x[1] == x[2], zip(a, b))

function is_stale!(cs::CacheState, u, t)
    cs.stale = any((!alleqeq(cs.u, u), cs.t_last_called ≠ t, !cs.called_in_IC))
    cs.called_in_IC = true
    if cs.stale
        cs.pscb_called[] = false
        cs.t_last_called = t
    end
    return cs.stale
end

function Wfact!(W, u, p, dtγ, t)
    println("inside Wfact!")
    @assert p.pscb_called[]
end
function T_lim!(uₜ, u, p, t)
    println("inside T_lim!")
    @assert p.pscb_called[]
end
function T_exp!(uₜ, u, p, t)
    println("inside T_exp!")
    @assert p.pscb_called[]
end
function T_imp!(uₜ, u, p, t)
    println("inside T_imp!")
    @assert p.pscb_called[]
end
function lim!(u, p, t, u_ref)
    println("inside lim!")
    @assert p.pscb_called[]
end
function apply_filter!(u, p, t, loc)
    println("inside apply_filter!: $loc")
    @assert p.pscb_called[]
end
post_explicit_stage_callback!(u, p, t, loc) = 
    post_stage_callback!(u, p, t, loc)
post_implicit_stage_callback!(u, p, t, loc) = 
    post_stage_callback!(u, p, t, loc)
function post_stage_callback!(u, p, t, loc)
    println("inside post_stage_callback!: $loc")
    # error("oops")
    is_stale!(p.cache_state, u, t) || return nothing
    if p.pscb_called[]
        error("post_stage_callback! too many times")
    end
    u .= rand()
    @. p.cache_state.u = u
    p.pscb_called[] = true
end

FT = Float64
t₀ = FT(0)
tspan = (t₀, FT(1))
dt = FT(0.1)
u₀ = FT[rand()]
pscb_called=Ref(false)
p = (;
    pscb_called,
    cache_state=CacheState(u₀, pscb_called, t₀),
)
struct SchurComplementW end
Base.similar(w::SchurComplementW) = w
import LinearAlgebra
function LinearAlgebra.ldiv!(x,A::SchurComplementW,b)
    println("inside LinearAlgebra.ldiv!")
end
import OrdinaryDiffEq as ODE

implicit_func=ODE.ODEFunction(
                T_imp!;
                Wfact=Wfact!,
                jac_prototype = SchurComplementW(),
                tgrad = (∂Y∂t, Y, p, t) -> (∂Y∂t .= 0),
            )
f = CTS.ClimaODEFunction(;
    T_lim!,T_exp!,T_imp! = implicit_func,
    lim!,apply_filter!,
    post_explicit_stage_callback!,
    post_implicit_stage_callback!
)

function some_callback1!(integrator)
    println("inside some_callback1!")
end
function some_callback2!(integrator)
    println("inside some_callback2!")
end

function get_dcb(cb!)
    return ODE.DiscreteCallback(
            (u, t, integrator) -> true,
            cb!;
            initialize = (cb, u, t, integrator) -> cb!(integrator),
            save_positions = (false, false),
        )
end
callback = ODE.CallbackSet(get_dcb(some_callback1!),get_dcb(some_callback2!))
prob = ODE.ODEProblem(f, u₀, tspan, p; dt)
integrator = ODE.init(
    prob,
    CTS.IMEXAlgorithm(CTS.ARS343(),
        CTS.NewtonsMethod(;max_iters=3)),
    p,
    tspan;
    callback
)
ODE.step!(integrator)

