module TestLearningCurves

using Test
using MLJBase
using MLJTuning
import ComputationalResources: CPU1, CPUProcesses, CPUThreads
using Distributed
@everywhere begin
    using Random
    Random.seed!(1234*myid()) 
    using ..Models
end
using ..TestUtilities

using ..Models
import Random.seed!
seed!(1234)

x1 = rand(100);
x2 = rand(100);
x3 = rand(100);
X = (x1=x1, x2=x2, x3=x3);
y = 2*x1 .+ 5*x2 .- 3*x3 .+ 0.2*rand(100);

@testset_accelerated "learning curves" accel (exclude=[CPUThreads],) begin
    atom = FooBarRegressor()
    ensemble = EnsembleModel(atom=atom, n=50)
    mach = machine(ensemble, X, y)
    r_lambda = range(ensemble, :(atom.lambda),
                     lower=0.0001, upper=0.1, scale=:log10)
    if accel == CPU1()
        curve = @test_logs((:info, r"No measure"),
                           (:info, r"Training"),
                           learning_curve!(mach; range=r_lambda,
                                           acceleration=accel))
    else
        curve = learning_curve!(mach; range=r_lambda,
                                acceleration=accel)
    end
    @test curve isa NamedTuple{(:parameter_name,
                                :parameter_scale,
                                :parameter_values,
                                :measurements)}
    @test length(curve.parameter_values) == length(curve.measurements)
    atom.lambda=0.3
    r_n = range(ensemble, :n, lower=10, upper=100)
    curves = learning_curve!(mach; range=r_n, n_curves=2, resolution=7)
    @test size(curves.measurements) == (length(curves.parameter_values), 2)
    @test length(curves.parameter_values) == 7

    # individual curves are different:
    @test !(curves.measurements[1,1] ≈ curves.measurements[1,2])
end

end # module
true
