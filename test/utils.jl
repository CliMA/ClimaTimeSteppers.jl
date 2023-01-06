import ClimaTimeSteppers as CTS
using Test

problem(test_case, tab::CTS.AbstractIMEXARKTableau) = test_case.split_prob
problem(test_case, tab) = test_case.prob
