import ClimaTimeSteppers as CTS
using Test

problem(test_case, tab::CTS.AbstractIMEXARKTableau) = test_case.split_prob
problem(test_case, tab::CTS.AbstractIMEXSSPARKTableau) = test_case.split_prob
problem(test_case, tab) = test_case.prob

algorithm(tab::CTS.AbstractIMEXARKTableau) = CTS.IMEXARKAlgorithm(tab, NewtonsMethod(; max_iters = 2))
algorithm(tab::CTS.AbstractIMEXSSPARKTableau) = CTS.IMEXSSPRKAlgorithm(tab, NewtonsMethod(; max_iters = 2))
algorithm(tab) = tab
