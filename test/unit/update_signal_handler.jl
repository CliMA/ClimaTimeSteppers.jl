using ClimaTimeSteppers, Test

@testset "UpdateSignalHandler" begin

    @testset "UpdateEvery" begin
        handler = UpdateEvery(NewTimeStep)

        # Triggers on matching signal type
        @test ClimaTimeSteppers.needs_update!(handler, NewTimeStep(0.0)) == true
        @test ClimaTimeSteppers.needs_update!(handler, NewTimeStep(1.0)) == true

        # Does not trigger on non-matching signal type
        @test ClimaTimeSteppers.needs_update!(handler, NewNewtonSolve()) == false
        @test ClimaTimeSteppers.needs_update!(handler, NewNewtonIteration()) == false
    end

    @testset "UpdateEveryN" begin
        handler = UpdateEveryN(3, NewNewtonIteration)

        # Triggers on iteration 0, then every 3rd
        @test ClimaTimeSteppers.needs_update!(handler, NewNewtonIteration()) == true  # counter=0 → true, inc to 1
        @test ClimaTimeSteppers.needs_update!(handler, NewNewtonIteration()) == false # counter=1 → false, inc to 2
        @test ClimaTimeSteppers.needs_update!(handler, NewNewtonIteration()) == false # counter=2 → false, inc to 0
        @test ClimaTimeSteppers.needs_update!(handler, NewNewtonIteration()) == true  # counter=0 → true, inc to 1
        @test ClimaTimeSteppers.needs_update!(handler, NewNewtonIteration()) == false

        # Non-matching signal does nothing
        @test ClimaTimeSteppers.needs_update!(handler, NewNewtonSolve()) == false
    end

    @testset "UpdateEveryDt" begin
        handler = UpdateEveryDt(0.5, Ref(true), Ref(0.0))

        # First call always triggers
        @test ClimaTimeSteppers.needs_update!(handler, NewTimeStep(0.0)) == true

        # Too soon — doesn't trigger
        @test ClimaTimeSteppers.needs_update!(handler, NewTimeStep(0.3)) == false

        # After dt elapsed — triggers
        @test ClimaTimeSteppers.needs_update!(handler, NewTimeStep(0.5)) == true

        # Too soon again
        @test ClimaTimeSteppers.needs_update!(handler, NewTimeStep(0.7)) == false

        # After dt elapsed from last update
        @test ClimaTimeSteppers.needs_update!(handler, NewTimeStep(1.0)) == true

        # Non-matching signal does nothing
        @test ClimaTimeSteppers.needs_update!(handler, NewNewtonSolve()) == false
        @test ClimaTimeSteppers.needs_update!(handler, NewNewtonIteration()) == false
    end

    @testset "DSS signal hierarchy" begin
        # Hierarchy: EndOfStep <: EndOfStage <: WithDSS <: UpdateSignal.
        @test EndOfStep <: EndOfStage <: WithDSS <: ClimaTimeSteppers.UpdateSignal

        # Each concrete singleton lives at its expected level.
        @test ClimaTimeSteppers.WithDSSSignal <: WithDSS
        @test ClimaTimeSteppers.EndOfStageSignal <: EndOfStage
        @test ClimaTimeSteppers.EndOfStepSignal <: EndOfStep

        # `UpdateEvery` uses subtype dispatch (`S <: U`): a broad subscription
        # matches any concrete signal on a narrower level.
        h_dss = UpdateEvery(WithDSS)
        h_stage = UpdateEvery(EndOfStage)
        h_step = UpdateEvery(EndOfStep)

        sig_dss = ClimaTimeSteppers.WithDSSSignal()
        sig_stage = ClimaTimeSteppers.EndOfStageSignal()
        sig_step = ClimaTimeSteppers.EndOfStepSignal()

        # WithDSS handler fires on any DSS-related signal.
        @test ClimaTimeSteppers.needs_update!(h_dss, sig_dss) == true
        @test ClimaTimeSteppers.needs_update!(h_dss, sig_stage) == true
        @test ClimaTimeSteppers.needs_update!(h_dss, sig_step) == true

        # EndOfStage handler skips the pre-implicit WithDSS-only fire.
        @test ClimaTimeSteppers.needs_update!(h_stage, sig_dss) == false
        @test ClimaTimeSteppers.needs_update!(h_stage, sig_stage) == true
        @test ClimaTimeSteppers.needs_update!(h_stage, sig_step) == true

        # EndOfStep handler fires only at end of step.
        @test ClimaTimeSteppers.needs_update!(h_step, sig_dss) == false
        @test ClimaTimeSteppers.needs_update!(h_step, sig_stage) == false
        @test ClimaTimeSteppers.needs_update!(h_step, sig_step) == true

        # DSS signals do not fire Newton-signal subscriptions and vice versa.
        h_newton = UpdateEvery(NewNewtonIteration)
        @test ClimaTimeSteppers.needs_update!(h_newton, sig_step) == false
        @test ClimaTimeSteppers.needs_update!(h_dss, NewNewtonIteration()) == false
    end
end
