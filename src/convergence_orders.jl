"""
    theoretical_convergence_order(::AbstractTableau, ::AbstractODEFunction)

Return the expected convergence order of the tableau for the given tendency
function (assuming that the method converges with this tendency).
"""
function theoretical_convergence_order end

function theoretical_convergence_order(tableau::AbstractIMEXARKTableau, tendency::ClimaODEFunction)
    (imp_order, exp_order, combined_order) = theoretical_imex_convergence_orders(tableau)
    has_imp = !isnothing(tendency.T_imp!)
    has_exp = !isnothing(tendency.T_exp!) || !isnothing(tendency.T_lim!)
    has_imp && !has_exp && return imp_order
    !has_imp && has_exp && return exp_order
    has_imp && has_exp && return combined_order
    return 0
end

"""
    theoretical_imex_convergence_orders(::AbstractIMEXARKTableau)

Return a tuple containing the expected convergence order of the tableau when
using only an implicit tendency, the order when using only an explicit tendency,
and the order when using both tendencies.
"""
function theoretical_imex_convergence_orders end

theoretical_imex_convergence_orders(::ARS111) = (1, 1, 1)
theoretical_imex_convergence_orders(::ARS121) = (1, 1, 1)
theoretical_imex_convergence_orders(::ARS122) = (2, 2, 2)
theoretical_imex_convergence_orders(::ARS222) = (2, 2, 2)
theoretical_imex_convergence_orders(::ARS232) = (2, 3, 2)
theoretical_imex_convergence_orders(::ARS233) = (3, 3, 3)
theoretical_imex_convergence_orders(::ARS343) = (3, 4, 3)
theoretical_imex_convergence_orders(::ARS443) = (3, 3, 3)
theoretical_imex_convergence_orders(::Union{IMKG232a, IMKG232b}) = (2, 2, 2)
theoretical_imex_convergence_orders(::Union{IMKG242a, IMKG242b}) = (2, 4, 2)
theoretical_imex_convergence_orders(::IMKG243a) = (2, 4, 2)
theoretical_imex_convergence_orders(::Union{IMKG252a, IMKG252b}) = (2, 2, 2)
theoretical_imex_convergence_orders(::Union{IMKG253a, IMKG253b}) = (2, 2, 2)
theoretical_imex_convergence_orders(::Union{IMKG254a, IMKG254b, IMKG254c}) = (2, 2, 2)
theoretical_imex_convergence_orders(::IMKG342a) = (3, 4, 3)
theoretical_imex_convergence_orders(::IMKG343a) = (3, 4, 3)
theoretical_imex_convergence_orders(::DBM453) = (3, 3, 3)
theoretical_imex_convergence_orders(::HOMMEM1) = (2, 3, 2)
theoretical_imex_convergence_orders(::SSP333c) = (3, 3, 3)
