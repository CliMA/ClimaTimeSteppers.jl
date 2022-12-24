import ClimaTimeSteppers as CTS
using Test

function problem_algo(test_case, tab)
    if tab() isa CTS.AbstractIMEXARKTableau
        max_iters = test_case.linear_implicit ? 1 : 2 # TODO: is 2 enough?
        alg = CTS.IMEXARKAlgorithm(tab(), NewtonsMethod(; max_iters))
        prob = test_case.split_prob
    else
        alg = tab()
        prob = test_case.prob
    end
    return (prob, alg)
end
