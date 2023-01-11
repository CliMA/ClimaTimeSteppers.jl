import ClimaTimeSteppers as CTS
using Test

problem(test_case, tab::CTS.IMEXAlgorithmName) = test_case.split_prob
problem(test_case, tab) = test_case.prob

algorithm(tab::CTS.IMEXAlgorithmName) = CTS.IMEXAlgorithm(tab, NewtonsMethod(; max_iters = 2))
algorithm(tab) = tab
