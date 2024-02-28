@inline function compute_T_lim_T_exp!(
    T_lim,
    T_exp,
    U,
    p,
    t,
    T_lim!,
    T_exp!,
    ::Union{Nothing, ClimaComms.AbstractCommsContext},
)
    T_lim!(T_lim, U, p, t)
    T_exp!(T_exp, U, p, t)
end
