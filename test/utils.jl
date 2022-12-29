import ClimaTimeSteppers as CTS
using Test

problem(test_case, tab::CTS.AbstractIMEXARKTableau) = test_case.split_prob
problem(test_case, tab) = test_case.prob

function algorithm(test_case, tab)
    return if tab isa CTS.AbstractIMEXARKTableau
        max_iters = test_case.linear_implicit ? 1 : 2 # TODO: is 2 enough?
        CTS.IMEXARKAlgorithm(tab, NewtonsMethod(; max_iters))
    else
        tab
    end
end
