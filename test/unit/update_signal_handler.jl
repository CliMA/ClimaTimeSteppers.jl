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
end
