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

function is_stale!(meta::CTS.AlgMeta, cs::CacheState, u, t)
    cs.stale = any((!alleqeq(cs.u, u), cs.t_last_called ≠ t, !cs.called_in_IC, meta.meta[2].meta == :newton_solve))
    cs.called_in_IC = true
    if cs.stale
        cs.pscb_called[] = false
        cs.t_last_called = t
    else
        println("Cache is re-useable: stale:$(cs.stale), !alleqeq=$(!alleqeq(cs.u, u)), t_last_called ≠ t: $(cs.t_last_called ≠ t), !called_in_IC: $(!cs.called_in_IC)")
    end
    return cs.stale
end
function Wfact!(meta::CTS.AlgMeta, W, u, p, dtγ, t)
    println("inside Wfact! $meta")
    @assert p.pscb_called[]
end
function T_lim!(meta::CTS.AlgMeta, uₜ, u, p, t)
    println("inside T_lim! $meta")
    @assert p.pscb_called[]
end
function T_exp!(meta::CTS.AlgMeta, uₜ, u, p, t)
    println("inside T_exp! $meta")
    @assert p.pscb_called[]
end
function T_imp!(meta::CTS.AlgMeta, uₜ, u, p, t)
    println("inside T_imp! $meta")
    @assert p.pscb_called[]
end
function lim!(meta::CTS.AlgMeta, u, p, t, u_ref)
    println("inside lim! $meta")
    # @assert p.pscb_called[]
    u .= rand() # imitate limiter
end
function apply_filter!(meta::CTS.AlgMeta, u, p, t)
    println("inside apply_filter! (dss): $meta")
    # @assert p.pscb_called[]
    u .= rand() # imitate dss
end

post_explicit_stage_callback!(meta::CTS.AlgMeta, u, p, t) = 
    post_stage_callback!(CTS.AlgMeta((:exp, meta)), u, p, t)

post_implicit_stage_callback!(meta::CTS.AlgMeta, u, p, t) = 
    post_stage_callback!(CTS.AlgMeta((:imp, meta)), u, p, t)

function post_stage_callback!(meta::CTS.AlgMeta, u, p, t)
    println("inside post_stage_callback! (set_precomputed_quantities!): $meta")
    is_stale!(meta, p.cache_state, u, t) || return nothing
    # println("running post_stage_callback!: $meta")
    if p.pscb_called[]
        error("post_stage_callback! too many times")
    end
    u .= rand()
    @. p.cache_state.u = u
    p.pscb_called[] = true
    p.pscb_counter[] += 1
end

FT = Float64
t₀ = FT(0)
tspan = (t₀, FT(1))
dt = FT(0.1)
u₀ = FT[rand()]
pscb_called=Ref(false)
pscb_counter=Ref(0)
p = (;
    pscb_called,
    pscb_counter,
    cache_state=CacheState(u₀, pscb_called, t₀),
)
struct SchurComplementW end
Base.similar(w::SchurComplementW) = w
import LinearAlgebra
function LinearAlgebra.ldiv!(x,A::SchurComplementW,b)
    println("inside LinearAlgebra.ldiv!")
end
import SciMLBase
import OrdinaryDiffEq as ODE

struct DebugMetaFunc{F} <: CTS.AbstractMetaFunc; f::F; end
DebugMetaFunc(x::Nothing) = x
CTS.meta_tuple(::DebugMetaFunc, meta) = (meta,)

implicit_func = SciMLBase.ODEFunction(
                DebugMetaFunc(T_imp!);
                Wfact=DebugMetaFunc(Wfact!),
                jac_prototype = SchurComplementW(),
                tgrad = (∂Y∂t, Y, p, t) -> (∂Y∂t .= 0),
            )

f = CTS.ClimaODEFunction(
    DebugMetaFunc(T_lim!),
    DebugMetaFunc(T_exp!),
    implicit_func,
    DebugMetaFunc(lim!),
    DebugMetaFunc(apply_filter!),
    DebugMetaFunc(post_explicit_stage_callback!),
    DebugMetaFunc(post_implicit_stage_callback!),
    CTS.DebugLogger()
)

function some_callback1!(integrator)
    println("inside some_callback1!")
    @assert p.pscb_called[]
end
function some_callback2!(integrator)
    println("inside some_callback2!")
    @assert p.pscb_called[]
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
        CTS.NewtonsMethod(;max_iters=1)),
    p,
    tspan;
    callback
)
ODE.step!(integrator)

@show integrator.p.pscb_counter

