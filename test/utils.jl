import ClimaTimeSteppers as CTS
using Test

problem(test_case, tab::CTS.ARKAlgorithmName) = test_case.split_prob
problem(test_case, tab) = test_case.prob

algorithm(tab::CTS.ARKAlgorithmName) = CTS.ARKAlgorithm(tab, NewtonsMethod(; max_iters = 2))
algorithm(tab::CTS.RosenbrockAlgorithmName) = CTS.RosenbrockAlgorithm(ClimaTimeSteppers.tableau(tab))
algorithm(tab) = tab
