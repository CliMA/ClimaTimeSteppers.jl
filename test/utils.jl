import ClimaTimeSteppers as CTS
using Test

problem(test_case, tab::CTS.AbstractIMEXARKTableau) = test_case.split_prob
problem(test_case, tab) = test_case.prob

function algorithm(tab)
    return if tab isa CTS.AbstractIMEXARKTableau
        CTS.IMEXARKAlgorithm(tab, NewtonsMethod(; max_iters = 2))
    else
        tab
    end
end
