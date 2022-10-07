#=
An s-stage (DIRK) IMEX ARK method for solving ∂u/∂t = f_exp(u, t) + f_imp(u, t)
is given by
    u_next :=
        u + Δt * ∑_{χ∈(exp,imp)} ∑_{i=1}^s b_χ[i] * f_χ(U[i], t + Δt * c_χ[i]),
    where
    U[i] :=
        u + Δt * ∑_{j=1}^{i-1} a_exp[i,j] * f_exp(U[j], t + Δt * c_exp[j]) +
        Δt * ∑_{j=1}^i a_imp[i,j] * f_imp(U[j], t + Δt * c_imp[j])
    ∀ i ∈ 1:s
Here, u_next denotes the value of u(t) at t_next = t + Δt.
The values a_χ[i,j] are called the "internal coefficients", the b_χ[i] are
called the "weights", and the c_χ[i] are called the "abcissae" (or "nodes").
The abscissae are often defined as c_χ[i] := ∑_{j=1}^s a_χ[i,j] for the explicit
and implicit methods to be "internally consistent", with c_exp[i] = c_imp[i] for
the overall IMEX method to be "internally consistent", but this is not required.
If the weights are defined as b_χ[j] := a_χ[s,j], then u_next = U[s]; i.e., the
method is FSAL (first same as last).

To simplify our notation, let
    a_χ[s+1,j] := b_χ[j] ∀ j ∈ 1:s,
    F_χ[j] := f_χ(U[j], t + Δt * c_χ[j]) ∀ j ∈ 1:s, and
    Δu_χ[i,j] := Δt * a_χ[i,j] * F_χ[j] ∀ i ∈ 1:s+1, j ∈ 1:s
This allows us to rewrite our earlier definitions as
    u_next = u + ∑_{χ∈(exp,imp)} ∑_{i=1}^s Δu_χ[s+1,i], where
    U[i] = u + ∑_{j=1}^{i-1} Δu_exp[i,j] + ∑_{j=1}^i Δu_imp[i,j] ∀ i ∈ 1:s

We will now rewrite the algorithm so that we can express each value of F_χ in
terms of the first increment Δu_χ that it is used to generate.
First, ∀ j ∈ 1:s, let
    first_i_χ[j] := min(i ∈ 1:s+1 ∣ a_χ[i,j] != 0)
Note that first_i_χ[j] is undefined if the j-th column of a_χ only contains zeros.
Also, note that first_i_imp[j] >= j and first_i_exp[j] > j ∀ j ∈ 1:s.
In addition, ∀ i ∈ 1:s+1, let
    new_js_χ[i] := [j ∈ 1:s ∣ first_i_χ[j] == i],
    old_js_χ[i] := [j ∈ 1:s ∣ first_i_χ[j] < i && a_χ[i,j] != 0], and
    N_χ[i] := length(new_js_χ[i])
We can then define, ∀ i ∈ 1:s+1,
    ũ[i] := u + ∑_{χ∈(exp,imp)} ∑_{j ∈ old_js_χ[i]} Δu_χ[i,j] and
    Û_χ[i,k] := Û_χ[i,k-1] + Δu_χ[i,new_js_χ[i][k]] ∀ k ∈ 1:N_χ[i], where
    Û_exp[i,0] := ũ[i] and Û_imp[i,0] := Û_exp[i,N_exp[i]]
We then find that
    u_next = Û_imp[s+1,N_imp[s+1]] and U[i] = Û_imp[i,N_imp[i]] ∀ i ∈ 1:s
Let
    all_js_χ := [j ∈ 1:s | isdefined(first_i_χ[j])]
Next, ∀ j ∈ all_js_χ, let
    K_χ[j] := k ∈ N_χ[first_i_χ[j]] | new_js_χ[first_i_χ[j]][k] == j
We then have that, ∀ j ∈ all_js_χ,
    Û_χ[first_i_χ[j],K_χ[j]] = Û_χ[first_i_χ[j],K_χ[j]-1] + Δu_χ[first_i_χ[j],j]
Since a_χ[first_i_χ[j],j] != 0, this means that, ∀ j ∈ all_js_χ,
    F_χ[j] = (Û_χ[first_i_χ[j],K_χ[j]] - Û_χ[first_i_χ[j],K_χ[j]-1]) / (Δt * a_χ[first_i_χ[j],j])

Now, suppose that we want to modify this algorithm so that we can apply a
filter/limiter during the addition of the increments Δu_χ[i,new_js_χ[i][k]].
Specifically, instead of specifying f_χ(u, t), we want to specify
g_χ(û, u, t, Δt) and redefine, ∀ i ∈ 1:s+1 and ∀ k ∈ 1:N_χ[i],
    Û_χ[i,k] :=
        g_χ(
            Û_χ[i,k-1],
            U[new_js_χ[i][k]],
            t + Δt * c_χ[new_js_χ[i][k]],
            Δt * a_χ[i,new_js_χ[i][k]]
        )
Note that specifying g_χ(û, u, t, Δt) := û + Δt * f_χ(u, t) is equivalent to not
using any filters/limiters.
We can use our earlier expression to redefine F_χ[j] as, ∀ j ∈ all_js_χ,
    F_χ[j] := (Û_χ[first_i_χ[j],K_χ[j]] - Û_χ[first_i_χ[j],K_χ[j]-1]) / (Δt * a_χ[first_i_χ[j],j])
We then have that, ∀ i ∈ 1:s+1 and ∀ j ∈ all_js_χ,
    Δu_χ[i,j] = ā_χ[i,j] * ΔÛ_χ[j], where
    ā_χ[i,j] := a_χ[i,j]/a_χ[first_i_χ[j],j] and
    ΔÛ_χ[j] := Û_χ[first_i_χ[j],K_χ[j]] - Û_χ[first_i_χ[j],K_χ[j]-1]
We can then use these values of Δu_χ[i,j] to determine each value of ũ[i].

Now, ∀ i ∈ 1:s+1, let
    Js_to_save_χ[i] := [j ∈ new_js_χ[i] | max(i′ ∈ 1:s+1 ∣ a_χ[i′,j] != 0) > i]
Note that we only need to compute F_χ[j] (or, rather, ΔÛ_χ[j]) if there is some
i ∈ 1:s+1 for which j ∈ Js_to_save_χ[i], since only then is there some value of
Δu_χ[i,j] that is computed based on F_χ[j].

This procedure of computing the values of F_χ (or, rather, the values of ΔÛ_χ)
from the values of Û_χ and using them to compute ũ[i] is rather inefficient, and
it would be better to directly use the values of Û_χ to compute ũ[i].
From the previous section, we know that, ∀ i ∈ 1:s+1,
    ũ[i] =
        u +
        ∑_{χ∈(exp,imp)} ∑_{j ∈ old_js_χ[i]} ā_χ[i,j] * (Û_χ[first_i_χ[j],K_χ[j]] - Û_χ[first_i_χ[j],K_χ[j]-1])
Now, ∀ i ∈ 1:s+1, let
    old_js1_χ[i] := [j ∈ old_js_χ[i] | K_χ[j] == 1] and
    old_js2_χ[i] := [j ∈ old_js_χ[i] | K_χ[j] > 1]
Since Û_exp[i,0] = ũ[i] and Û_imp[i,0] = Û_exp[i,N_exp[i]], we then have that
    ũ[i] =
        u +
        ∑_{j ∈ old_js1_exp[i]} ā_exp[i,j] * (Û_exp[first_i_exp[j],1] - ũ[first_i_exp[j]]) +
        ∑_{j ∈ old_js1_imp[i]} ā_imp[i,j] * (Û_imp[first_i_imp[j],1] - Û_exp[first_i_imp[j],N_exp[first_i_imp[j]]]) +
        ∑_{χ∈(exp,imp)} ∑_{j ∈ old_js2_χ[i]} ā_χ[i,j] * (Û_χ[first_i_χ[j],K_χ[j]] - Û_χ[first_i_χ[j],K_χ[j]-1])
Next, ∀ i ∈ 1:s+1, let
    old_js11_imp[i] := [j ∈ old_js1_imp[i] | N_exp[first_i_imp[j]] == 0] and
    old_js12_imp[i] := [j ∈ old_js1_imp[i] | N_exp[first_i_imp[j]] > 0]
Since Û_exp[i,0] = ũ[i], this means that
    ũ[i] =
        u +
        ∑_{j ∈ old_js1_exp[i]} ā_exp[i,j] * (Û_exp[first_i_exp[j],1] - ũ[first_i_exp[j]]) +
        ∑_{j ∈ old_js11_imp[i]} ā_imp[i,j] * (Û_imp[first_i_imp[j],1] - ũ[first_i_imp[j]]) +
        ∑_{j ∈ old_js12_imp[i]} ā_imp[i,j] * (Û_imp[first_i_imp[j],1] - Û_exp[first_i_imp[j],N_exp[first_i_imp[j]]]) +
        ∑_{χ∈(exp,imp)} ∑_{j ∈ old_js2_χ[i]} ā_χ[i,j] * (Û_χ[first_i_χ[j],K_χ[j]] - Û_χ[first_i_χ[j],K_χ[j]-1])
We will now show that, ∀ i ∈ 1:s+1, there are some Q₀ and Q_χ such that
    ũ[i] = Q₀[i] * u + ∑_{χ∈(exp,imp)} ∑_{j=1}^{i-1} ∑_{k=1}^{N_χ[j]} Q_χ[i, j, k] * Û_χ[j, k]
First, we check the base case: ũ[1] = u, so that
    ũ[1] = Q₀[1] * u, where Q₀[1] = 1
Next, we apply the inductive step...
Is this too messy to do in the general case?

Don't forget about the possible memory optimizations!
=#
export IMEXARKAlgorithm, make_IMEXARKAlgorithm

using Base: broadcasted, materialize!
using StaticArrays: SMatrix, SVector

"""
    IMEXARKAlgorithm <: DistributedODEAlgorithm

A generic implementation of an IMEX ARK algorithm that can handle arbitrary
Butcher tableaus and problems specified using either `ForwardEulerODEFunction`s
or regular `ODEFunction`s.
"""
struct IMEXARKAlgorithm{as, cs, N} <: DistributedODEAlgorithm
    newtons_method::N
end

IMEXARKAlgorithm{as, cs}(newtons_method::N) where {as, cs, N} =
    IMEXARKAlgorithm{as, cs, N}(newtons_method)

"""
    make_IMEXARKAlgorithm(; a_exp, b_exp, c_exp, a_imp, b_imp, c_imp)

Generates an `IMEXARKAlgorithm` type from an IMEX ARK Butcher tableau. Only
`a_exp` and `a_imp` are required arguments; the default values for `b_exp` and
`b_imp` assume that the algorithm is FSAL (first same as last), and the default
values for `c_exp` and `c_imp` assume that the algorithm is internally
consistent.
"""
function make_IMEXARKAlgorithm(;
    a_exp::SMatrix{s, s},
    b_exp::SVector{s} = vec(a_exp[end, :]),
    c_exp::SVector{s} = vec(sum(a_exp; dims = 2)),
    a_imp::SMatrix{s, s},
    b_imp::SVector{s} = vec(a_imp[end, :]),
    c_imp::SVector{s} = vec(sum(a_imp; dims = 2)),
) where {s}
    @assert all(i -> all(j -> a_exp[i, j] == 0, i:s), 1:s)
    @assert all(i -> all(j -> a_imp[i, j] == 0, (i + 1):s), 1:s)
    if a_exp[end, :] == b_exp && a_imp[end, :] == b_imp
        as = (a_exp, a_imp)
    else
        as = (vcat(a_exp, b_exp'), vcat(a_imp, b_imp'))
    end
    cs = (c_exp, c_imp)
    return IMEXARKAlgorithm{as, cs}
end

# General helper functions
is_increment(f_type) = f_type <: ForwardEulerODEFunction
i_range(a) = Tuple(1:size(a, 1))
j_range(a) = Tuple(1:size(a, 2))
u_alias_is(a_exp, a_imp) = filter(
    i -> all(j -> a_exp[i, j] == a_imp[i, j] == 0, j_range(a_exp)),
    i_range(a_exp),
)

# Helper functions for increments
first_i(j, a) = findfirst(i -> a[i, j] != 0, i_range(a))
new_js(i, a) = filter(j -> first_i(j, a) == i, j_range(a))
js_to_save(i, a) = filter(
    j -> findlast(i′ -> a[i′, j] != 0, i_range(a)) > i,
    new_js(i, a),
)

# Helper functions for tendencies
has_implicit_step(i, a) = i <= size(a, 2) && a[i, i] != 0
save_tendency(i, a) = 
    !isnothing(findlast(i′ -> a[i′, i] != 0, (i + 1):size(a, 1)))

# Helper functions for increments and tendencies
old_js(i, a, f_type) = filter(
    j -> (
            is_increment(f_type) ?
            !isnothing(first_i(j, a)) && first_i(j, a) < i : j < i
        ) && a[i, j] != 0,
    j_range(a),
)

struct IMEXARKCache{as, cs, C, N}
    _cache::C
    newtons_method_cache::N
end

# TODO: Minimize allocations by finding a minimum vertex coloring of the
#       interval graph for all required cached values.
function cache(
    prob::DiffEqBase.AbstractODEProblem,
    alg::IMEXARKAlgorithm{as, cs};
    kwargs...
) where {as, cs}
    f_cache(χ, a, f_type) = is_increment(f_type) ?
        map(
            j -> Symbol(:ΔÛ, χ, :_, j) => similar(u),
            Iterators.flatten(map(i -> js_to_save(i, a), i_range(a))),
        ) :
        map(
            i -> Symbol(:f, χ, :_, i) => similar(u),
            filter(i -> save_tendency(i, a), i_range(a)),
        )
    u = prob.u0
    Uis = map(
        i -> Symbol(:U, i) => similar(u),
        filter(i -> !(i in u_alias_is(as[1], as[2])), i_range(as[1])[1:end - 1])
    )
    _cache = NamedTuple((
        :U_temp => similar(u),
        Uis...,
        f_cache(:exp, as[1], typeof(prob.f.f2))...,
        f_cache(:imp, as[2], typeof(prob.f.f1))...,
    ))
    newtons_method_cache =
        allocate_cache(alg.newtons_method, u, prob.f.f1.jac_prototype)
    return IMEXARKCache{as, cs, typeof(_cache), typeof(newtons_method_cache)}(
        _cache,
        newtons_method_cache,
    )
end

# Workarounds for not being allowed to use closures in a generated function
struct ImplicitError{F, U, P, T}
    ode_f!::F
    û::U
    p::P
    t::T
    Δt::T
end
struct ImplicitErrorJacobian{W, P, T}
    Wfact!::W
    p::P
    t::T
    Δt::T
end

(implicit_error::ImplicitError)(f, u) =
    implicit_error(f, u, implicit_error.ode_f!)
function ((; û, p, t, Δt)::ImplicitError)(f, u, ode_f!::ForwardEulerODEFunction)
    f .= û
    ode_f!(f, u, p, t, Δt)
    f .-= u
end
function ((; û, p, t, Δt)::ImplicitError)(f, u, ode_f!)
    ode_f!(f, u, p, t)
    f .= û .+ Δt .* f .- u
end
((; Wfact!, p, t, Δt)::ImplicitErrorJacobian)(j, u) = Wfact!(j, u, p, Δt, t)

function step_u_expr(
    ::Type{<:IMEXARKCache{as, cs}},
    ::Type{f_exp_type},
    ::Type{f_imp_type},
    ::Type{FT},
) where {as, cs, f_exp_type, f_imp_type, FT}
    function Δu_expr(i, j, χ, a, f_type)
        if is_increment(f_type)
            ΔÛj = :(_cache.$(Symbol(:ΔÛ, χ, :_, j)))
            return :(broadcasted(*, $(FT(a[i, j] / a[first_i(j, a), j])), $ΔÛj))
        else
            fj = :(_cache.$(Symbol(:f, χ, :_, j)))
            return :(broadcasted(*, dt * $(FT(a[i, j])), $fj))
        end
    end
    Δu_exprs(i, χ, a, f_type) =
        map(j -> Δu_expr(i, j, χ, a, f_type), old_js(i, a, f_type))

    χs = (:exp, :imp)
    fs = (:f2, :f1)
    f_types = (f_exp_type, f_imp_type)

    expr = :(
        (; broadcasted, materialize!) = Base;
        (; u, p, t, dt, sol, alg) = integrator;
        (; f) = sol.prob;
        (; f1, f2) = f;
        (; newtons_method) = alg;
        (; _cache, newtons_method_cache) = cache
    )

    is = i_range(as[1]) # or as[2]
    for i in is
        if i in u_alias_is(as[1], as[2])
            Ui = :u
        else
            Ui = i == is[end] ? :u : :(_cache.$(Symbol(:U, i)))
            all_Δu_exprs = (
                Δu_exprs(i, χs[1], as[1], f_types[1])...,
                Δu_exprs(i, χs[2], as[2], f_types[2])...,
            )
            ũi_expr = length(all_Δu_exprs) == 0 ? :u :
                :(broadcasted(+, u, $(all_Δu_exprs...)))
            expr = :($(expr.args...); materialize!($Ui, $ũi_expr))
            for (χ, a, c, f, f_type) in zip(χs, as, cs, fs, f_types)
                if is_increment(f_type)
                    for j in new_js(i, a)
                        Ûik_expr = :(
                            t′ = t + dt * $(FT(c[j]));
                            Δt′ = dt * $(FT(a[i, j]))
                        )
                        if j == i
                            Ûik_expr = :(
                                $(Ûik_expr.args...);
                                _cache.U_temp .= $Ui;
                                run!(
                                    newtons_method,
                                    newtons_method_cache,
                                    _cache.U_temp,
                                    ImplicitError($f, $Ui, p, t′, Δt′),
                                    ImplicitErrorJacobian($f.Wfact, p, t′, Δt′),
                                );
                                $Ui .= _cache.U_temp
                            )
                        else # this is why we store Uj
                            Uj = j in u_alias_is(as[1], as[2]) ? :u :
                                :(_cache.$(Symbol(:U, j)))
                            Ûik_expr = :(
                                $(Ûik_expr.args...);
                                $f($Ui, $Uj, p, t′, Δt′)
                            )
                        end
                        if j in js_to_save(i, a)
                            ΔÛj = :(_cache.$(Symbol(:ΔÛ, χ, :_, j)))
                            Ûik_expr = :(
                                $ΔÛj .= $Ui;
                                $(Ûik_expr.args...);
                                $ΔÛj .= $Ui .- $ΔÛj
                            )
                        end
                        expr = :($(expr.args...); $(Ûik_expr.args...))
                    end
                elseif has_implicit_step(i, a)
                    if save_tendency(i, a)
                        fi = :(_cache.$(Symbol(:f, χ, :_, i)))
                        save_tendency_expr =
                            :($fi .= (_cache.U_temp .- $Ui) ./ Δt′)
                    else
                        save_tendency_expr = :()
                    end
                    expr = :(
                        $(expr.args...);
                        t′ = t + dt * $(FT(c[i]));
                        Δt′ = dt * $(FT(a[i, i]));
                        _cache.U_temp .= $Ui;
                        run!(
                            newtons_method,
                            newtons_method_cache,
                            _cache.U_temp,
                            ImplicitError($f, $Ui, p, t′, Δt′),
                            ImplicitErrorJacobian($f.Wfact, p, t′, Δt′),
                        );
                        $save_tendency_expr;
                        $Ui .= _cache.U_temp
                    )
                end
            end
        end
        for (χ, a, c, f, f_type) in zip(χs, as, cs, fs, f_types)
            if !is_increment(f_type) && !has_implicit_step(i, a) &&
                save_tendency(i, a)
                fi = :(_cache.$(Symbol(:f, χ, :_, i)))
                expr = :(
                    $(expr.args...);
                    t′ = t + dt * $(FT(c[i]));
                    $f($fi, $Ui, p, t′);
                )
            end
        end
    end
    return :($(expr.args...); return u)
end

step_u!(integrator, cache::IMEXARKCache) =
    imex_ark_step_u!(
        integrator,
        cache,
        typeof(integrator.sol.prob.f.f2),
        typeof(integrator.sol.prob.f.f1),
        typeof(integrator.dt),
    )
@generated imex_ark_step_u!(
    integrator,
    cache,
    ::Type{f_exp_type},
    ::Type{f_imp_type},
    ::Type{FT},
) where {f_exp_type, f_imp_type, FT} =
    step_u_expr(cache, f_exp_type, f_imp_type, FT)

################################################################################

function not_generated_cache(
    prob::DiffEqBase.AbstractODEProblem,
    alg::IMEXARKAlgorithm{as, cs};
    kwargs...
) where {as, cs}
    f_cache(χ, a, f_type) = is_increment(f_type) ?
        map(
            j -> Symbol(:ΔÛ, χ, :_, j) => similar(u),
            Iterators.flatten(map(i -> js_to_save(i, a), i_range(a))),
        ) :
        map(
            i -> Symbol(:f, χ, :_, i) => similar(u),
            filter(i -> save_tendency(i, a), i_range(a)),
        )

    u = prob.u0
    Uis = map(
        i -> Symbol(:U, i) => similar(u),
        filter(i -> !(i in u_alias_is(as[1], as[2])), i_range(as[1])[1:end - 1])
    )
    _cache = NamedTuple((
        :U_temp => similar(u),
        Uis...,
        f_cache(:exp, as[1], typeof(prob.f.f2))...,
        f_cache(:imp, as[2], typeof(prob.f.f1))...,
    ))
    newtons_method_cache =
        allocate_cache(alg.newtons_method, u, prob.f.f1.jac_prototype)
    
    f_types = (typeof(prob.f.f2), typeof(prob.f.f1))
    @inbounds _cache = (;
        _cache..., 
        u_alias_is_ = u_alias_is(as[1], as[2]),
        first_i_s = map(χ -> map(j -> first_i(j, as[χ]), j_range(as[χ])), Tuple(1:2)),
        new_js_s = map(χ -> map(i -> new_js(i, as[χ]), i_range(as[χ])), Tuple(1:2)),
        js_to_save_s = map(χ -> map(i -> js_to_save(i, as[χ]), i_range(as[χ])), Tuple(1:2)),
        has_implicit_step_s = map(χ -> map(i -> has_implicit_step(i, as[χ]), i_range(as[χ])), Tuple(1:2)),
        save_tendency_s = map(χ -> map(i -> save_tendency(i, as[χ]), i_range(as[χ])), Tuple(1:2)),
        old_js_s = map(χ -> map(i -> old_js(i, as[χ], f_types[χ]), i_range(as[χ])), Tuple(1:2)),
    )

    return IMEXARKCache{as, cs, typeof(_cache), typeof(newtons_method_cache)}(
        _cache,
        newtons_method_cache,
    )
end
function not_generated_step_u!(integrator, cache::IMEXARKCache{as, cs}) where {as, cs}
    @inbounds begin
        (; u, p, t, dt, sol, alg) = integrator
        (; f) = sol.prob
        (; f1, f2) = f
        (; newtons_method) = alg
        (; _cache, newtons_method_cache) = cache

        FT = typeof(integrator.dt)
        χs = (:exp, :imp)
        fs = (f2, f1)
        f_types = (typeof(f2), typeof(f1))
        (; u_alias_is_, first_i_s, new_js_s, js_to_save_s, has_implicit_step_s, save_tendency_s, old_js_s) = _cache

        function Δu_broadcast(i, j, χ, a, f_type, first_i_)
            if is_increment(f_type)
                ΔÛj = getproperty(_cache, Symbol(:ΔÛ, χ, :_, j))
                return broadcasted(*, FT(a[i, j] / a[first_i_[j], j]), ΔÛj)
            else
                fj = getproperty(_cache, Symbol(:f, χ, :_, j))
                return broadcasted(*, dt * FT(a[i, j]), fj)
            end
        end
        Δu_broadcasts(i, χ, a, f_type, first_i_, old_js_) =
            map(j -> Δu_broadcast(i, j, χ, a, f_type, first_i_), old_js_[i])

        is = i_range(as[1])
        for i in is
            if i in u_alias_is_
                Ui = u
            else
                Ui = i == is[end] ? u : getproperty(_cache, Symbol(:U, i))
                all_Δu_broadcasts = (
                    Δu_broadcasts(i, χs[1], as[1], f_types[1], first_i_s[1], old_js_s[1])...,
                    Δu_broadcasts(i, χs[2], as[2], f_types[2], first_i_s[2], old_js_s[2])...,
                )
                ũi_broadcast = length(all_Δu_broadcasts) == 0 ? u :
                    broadcasted(+, u, all_Δu_broadcasts...)
                materialize!(Ui, ũi_broadcast)
                for index in 1:2
                    (χ, a, c, f, f_type, new_js_, js_to_save_, has_implicit_step_, save_tendency_) =
                        (χs[index], as[index], cs[index], fs[index], f_types[index], new_js_s[index], js_to_save_s[index], has_implicit_step_s[index], save_tendency_s[index])
                    if is_increment(f_type)
                        for j in new_js_[i]
                            t′ = t + dt * FT(c[j])
                            Δt′ = dt * FT(a[i, j])
                            if j in js_to_save_[i]
                                ΔÛj = getproperty(_cache, Symbol(:ΔÛ, χ, :_, j))
                                ΔÛj .= Ui
                            end
                            if j == i
                                _cache.U_temp .= Ui
                                run!(
                                    newtons_method,
                                    newtons_method_cache,
                                    _cache.U_temp,
                                    ImplicitError(f, Ui, p, t′, Δt′),
                                    ImplicitErrorJacobian(f.Wfact, p, t′, Δt′),
                                );
                                Ui .= _cache.U_temp
                            else # this is why we store Uj
                                Uj = j in u_alias_is_ ? u :
                                    getproperty(_cache, Symbol(:U, j))
                                f(Ui, Uj, p, t′, Δt′)
                            end
                            if j in js_to_save_[i]
                                ΔÛj .= Ui .- ΔÛj
                            end
                        end
                    elseif has_implicit_step_[i]
                        t′ = t + dt * FT(c[i])
                        Δt′ = dt * FT(a[i, i])
                        _cache.U_temp .= Ui
                        run!(
                            newtons_method,
                            newtons_method_cache,
                            _cache.U_temp,
                            ImplicitError(f, Ui, p, t′, Δt′),
                            ImplicitErrorJacobian(f.Wfact, p, t′, Δt′),
                        )
                        if save_tendency_[i]
                            fi = getproperty(_cache, Symbol(:f, χ, :_, i))
                            fi .= (_cache.U_temp .- Ui) ./ Δt′
                        end
                        Ui .= _cache.U_temp
                    end
                end
            end
            for index in 1:2
                (χ, c, f, f_type, has_implicit_step_, save_tendency_) =
                    (χs[index], cs[index], fs[index], f_types[index], has_implicit_step_s[index], save_tendency_s[index])
                if !is_increment(f_type) && !has_implicit_step_[i] &&
                    save_tendency_[i]
                    fi = getproperty(_cache, Symbol(:f, χ, :_, i))
                    t′ = t + dt * FT(c[i])
                    f(fi, Ui, p, t′)
                end
            end
        end
        return u
    end
end
