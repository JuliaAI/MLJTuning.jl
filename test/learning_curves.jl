module TestLearningCurves

using Test
using Distributed

@everywhere begin
    using MLJBase
    using MLJTuning
    using ..Models
    import ComputationalResources: CPU1, CPUProcesses, CPUThreads
    #using Distributed
end
using Random
Random.seed!(1234*myid())
using ..TestUtilities

x1 = rand(100);
x2 = rand(100);
x3 = rand(100);
X = (x1=x1, x2=x2, x3=x3);
y = 2*x1 .+ 5*x2 .- 3*x3 .+ 0.2*rand(100);

@testset_accelerated "learning curves" accel begin
    atom = FooBarRegressor()
    ensemble = EnsembleModel(atom=atom, n=50)
    mach = machine(ensemble, X, y)
    r_lambda = range(ensemble, :(atom.lambda),
                     lower=0.0001, upper=0.1, scale=:log10)
    printstyled("\n Testing progressmeter rngs option with $(accel) and CPU1 grid \n", color=:bold)
    if accel == CPU1()
        curve = @test_logs((:info, r"No measure"),
                           (:info, r"Training"),
                           (:info, r"Attempting"),
                           learning_curve(mach; range=r_lambda,
                                          acceleration=accel))
    else
        curve = learning_curve(mach; range=r_lambda,
                               acceleration=accel)
    end
    @test curve isa NamedTuple{(:parameter_name,
                                :parameter_scale,
                                :parameter_values,
                                :measurements)}
    @test length(curve.parameter_values) == length(curve.measurements)
    atom.lambda=0.3
    r_n = range(ensemble, :n, lower=10, upper=100)
    curves = learning_curve(mach; range=r_n, resolution=7,
                             acceleration=accel,
                             rngs = MersenneTwister.(1:3),
                             rng_name=:rng, verbosity = 1)
    @test size(curves.measurements) == (length(curves.parameter_values), 3)
    @test length(curves.parameter_values) == 7

    # individual curves are different:
    @test !(curves.measurements[1,1] ≈ curves.measurements[1,2])
    @test !(curves.measurements[1,1] ≈ curves.measurements[1,3])

    # reproducibility:
    curves2 = learning_curve(mach; range=r_n, resolution=7,
                             acceleration=accel,
                             rngs = 3,
                             rng_name=:rng, verbosity=0)
    @test curves2.measurements ≈ curves.measurements

    # alternative signature:
    curves3 = learning_curve(ensemble, X, y; range=r_n, resolution=7,
                             acceleration=accel,
                             rngs = 3,
                             rng_name=:rng, verbosity=0)

    @test curves2.measurements ≈ curves3.measurements

   # restricting rows gives different answer:
    curves4 = learning_curve(ensemble, X, y; range=r_n, resolution=7,
                             rows = 1:60,
                             acceleration=accel,
                             rngs = 3,
                             rng_name=:rng, verbosity=0)

    @test !(curves4.measurements[1] ≈ curves2.measurements[1])

end

@static if VERSION >= v"1.3.0-DEV.573"

@testset_accelerated "learning curves (CPUThreads grid) " accel begin
    atom = FooBarRegressor()
    ensemble = EnsembleModel(atom=atom, n=50)
    mach = machine(ensemble, X, y)
    r_lambda = range(ensemble, :(atom.lambda),
                     lower=0.0001, upper=0.1, scale=:log10)
    printstyled("\n Testing progressmeter rngs option with $(accel) and CPUThreads grid \n", color=:bold)
    if accel == CPU1()
        curve = @test_logs((:info, r"No measure"),
                           (:info, r"Training"),
                           (:info, r"Attempting"),
                           learning_curve(mach; range=r_lambda,
                                          acceleration=accel, acceleration_grid = CPUThreads()))
    else
        curve = learning_curve(mach; range=r_lambda,
                               acceleration=accel, acceleration_grid = CPUThreads())
    end
    @test curve isa NamedTuple{(:parameter_name,
                                :parameter_scale,
                                :parameter_values,
                                :measurements)}
    @test length(curve.parameter_values) == length(curve.measurements)
    atom.lambda=0.3
    r_n = range(ensemble, :n, lower=10, upper=100)
    curves = learning_curve(mach; range=r_n, resolution=7,
                             acceleration=accel, acceleration_grid = CPUThreads(),
                             rngs = MersenneTwister.(1:3),
                             rng_name=:rng, verbosity = 1)
    @test size(curves.measurements) == (length(curves.parameter_values), 3)
    @test length(curves.parameter_values) == 7

    # individual curves are different:
    @test !(curves.measurements[1,1] ≈ curves.measurements[1,2])
    @test !(curves.measurements[1,1] ≈ curves.measurements[1,3])

    # reproducibility:
    curves2 = learning_curve(mach; range=r_n, resolution=7,
                             acceleration=accel, acceleration_grid = CPUThreads(),
                             rngs = 3,
                             rng_name=:rng, verbosity=0)
    @test curves2.measurements ≈ curves.measurements

    # alternative signature:
    curves3 = learning_curve(ensemble, X, y; range=r_n, resolution=7,
                             acceleration=accel, acceleration_grid = CPUThreads(),
                             rngs = 3,
                             rng_name=:rng, verbosity=0)

    @test curves2.measurements ≈ curves3.measurements

    # restricting rows gives different answer:
    curves4 = learning_curve(ensemble, X, y; range=r_n, resolution=7,
                             acceleration=accel, acceleration_grid = CPUThreads(),
                             rows = 1:60,
                             rngs = 3,
                             rng_name=:rng, verbosity=0)

    @test !(curves4.measurements[1] ≈ curves2.measurements[1])

end
end

@testset_accelerated "learning curves (CPUProcesses grid) " accel begin
    atom = FooBarRegressor()
    ensemble = EnsembleModel(atom=atom, n=50)
    mach = machine(ensemble, X, y)
    r_lambda = range(ensemble, :(atom.lambda),
                     lower=0.0001, upper=0.1, scale=:log10)
    printstyled("\n Testing progressmeter rngs option with $(accel) and CPUProcesses grid \n", color=:bold)
    if accel == CPU1()
        curve = @test_logs((:info, r"No measure"),
                           (:info, r"Training"),
                           (:info, r"Attempting"),
                           learning_curve(mach; range=r_lambda,
                                          acceleration=accel, acceleration_grid = CPUProcesses()))
    else
        curve = learning_curve(mach; range=r_lambda,
                               acceleration=accel, acceleration_grid = CPUProcesses())
    end
    @test curve isa NamedTuple{(:parameter_name,
                                :parameter_scale,
                                :parameter_values,
                                :measurements)}
    @test length(curve.parameter_values) == length(curve.measurements)
    atom.lambda=0.3
    r_n = range(ensemble, :n, lower=10, upper=100)
    curves = learning_curve(mach; range=r_n, resolution=7,
                             acceleration=accel, acceleration_grid = CPUProcesses(),
                             rngs = MersenneTwister.(1:3),
                             rng_name=:rng, verbosity = 1)
    @test size(curves.measurements) == (length(curves.parameter_values), 3)
    @test length(curves.parameter_values) == 7

    # individual curves are different:
    @test !(curves.measurements[1,1] ≈ curves.measurements[1,2])
    @test !(curves.measurements[1,1] ≈ curves.measurements[1,3])

    # reproducibility:
    curves2 = learning_curve(mach; range=r_n, resolution=7,
                             acceleration=accel, acceleration_grid = CPUProcesses(),
                             rngs = 3,
                             rng_name=:rng, verbosity=0)
    @test curves2.measurements ≈ curves.measurements

    # alternative signature:
    curves3 = learning_curve(ensemble, X, y; range=r_n, resolution=7,
                             acceleration=accel, acceleration_grid = CPUProcesses(),
                             rngs = 3,
                             rng_name=:rng, verbosity=0)

    @test curves2.measurements ≈ curves3.measurements

    # restricting rows gives different answer:
    curves4 = learning_curve(ensemble, X, y; range=r_n, resolution=7,
                             acceleration=accel, acceleration_grid = CPUProcesses(),
                             rows=1:60,
                             rngs = 3,
                             rng_name=:rng, verbosity=0)

    @test !(curves4.measurements[1] ≈ curves2.measurements[1])

end


end # module
true
