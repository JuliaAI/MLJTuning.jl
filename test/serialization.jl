module TestSerialization

using Test
using MLJBase
using Serialization
using MLJTuning
using ..Models

function test_args(mach)
    # Check source nodes are empty if any
    for arg in mach.args
        if arg isa Source
            @test arg == source()
        end
    end
end

test_data(mach) = @test all([:old_rows, :data, :resampled_data, :cache]) do field
    !isdefined(mach, field) || isnothing(getfield(mach, field))
end

function generic_tests(mach₁, mach₂)
    test_args(mach₂)
    test_data(mach₂)
    @test mach₂.state == -1
    for field in (:frozen, :model, :old_model, :old_upstream_state)
        @test getfield(mach₁, field) == getfield(mach₂, field)
    end
end


@testset "Test TunedModel" begin
    filename = "tuned_model.jls"
    X, y = make_regression(100)
    base_model = DecisionTreeRegressor()
    tuned_model = TunedModel(
        model=base_model,
        tuning=Grid(),
        range=[range(base_model, :min_samples_split, values=[2,3,4])],
    )
    mach = machine(tuned_model, X, y)
    fit!(mach, rows=1:50, verbosity=0)
    smach = MLJBase.serializable(mach)
    @test smach.fitresult isa Machine
    @test smach.report == mach.report
    generic_tests(mach, smach)

    Serialization.serialize(filename, smach)
    smach = Serialization.deserialize(filename)
    MLJBase.restore!(smach)

    @test MLJBase.predict(smach, X) == MLJBase.predict(mach, X)
    @test fitted_params(smach) isa NamedTuple
    @test report(smach) == report(mach)

    rm(filename)

    # End to end
    MLJBase.save(filename, mach)
    smach = machine(filename)
    @test predict(smach, X) == predict(mach, X)

    rm(filename)

end

end

true
