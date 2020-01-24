module TestTunedModels

using Distributed

using Test
using MLJTuning
using MLJBase
import ComputationalResources: CPU1, CPUProcesses, CPUThreads
using Random
Random.seed!(1234)
@everywhere using ..Models
using ..TestUtilities

N = 30
x1 = rand(N);
x2 = rand(N);
x3 = rand(N);
X = (x1=x1, x2=x2, x3=x3);
y = 2*x1 .+ 5*x2 .- 3*x3 .+ 0.4*rand(N);

m(K) = KNNRegressor(K=K)
r = [m(K) for K in 2:13]

@testset "constructor" begin
    @test_throws ErrorException TunedModel(model=first(r), tuning=Explicit(),
                                           measure=rms)
    @test_throws ErrorException TunedModel(tuning=Explicit(),
                                           range=r, measure=rms)
    @test_throws ErrorException TunedModel(model=42, tuning=Explicit(),
                                           range=r, measure=rms)
    @test_logs((:info, r"No measure specified"),
               TunedModel(model=first(r), tuning=Explicit(), range=r))
end

results = [(evaluate(model, X, y,
                     resampling=CV(nfolds=2),
                     measure=rms,
                     verbosity=0,)).measurement[1] for model in r]

@testset "measure compatibility check" begin
    tm = TunedModel(model=first(r), tuning=Explicit(),
                    range=r, resampling=CV(nfolds=2),
                    measures=cross_entropy)
    @test_throws ArgumentError fit(tm, 0, X, y)
end

@testset_accelerated "basic fit" accel (exclude=[CPUThreads],) begin
    best_index = argmin(results)
    tm = TunedModel(model=first(r), tuning=Explicit(),
                    range=r, resampling=CV(nfolds=2),
                    measures=[rms, l1], acceleration=accel)
    fitresult, meta_state, report = fit(tm, 0, X, y);
    history, _, state = meta_state;
    results2 = map(event -> last(event).measurement[1], history)
    @test results2 ≈ results
    @test fitresult.model == collect(r)[best_index]
    @test report.best_model == collect(r)[best_index]
    @test report.history == history
end

@testset_accelerated "accel. resampling" accel (exclude=[CPUThreads],) begin
    tm = TunedModel(model=first(r), tuning=Explicit(),
                    range=r, resampling=CV(nfolds=2),
                    measures=[rms, l1], acceleration_resampling=accel)
    fitresult, meta_state, report = fit(tm, 0, X, y);
    history, _, state = meta_state;
    results3 = map(event -> last(event).measurement[1], history)
    @test results3 ≈ results
end

@testset_accelerated("under/over supply of models", accel,
                     (exclude=[CPUThreads],), begin
                     tm = TunedModel(model=first(r), tuning=Explicit(),
                                     range=r, measures=[rms, l1],
                                     acceleration=accel,
                                     resampling=CV(nfolds=2),
                                     n=4)
    mach = machine(tm, X, y)
    fit!(mach, verbosity=0)
    history = MLJBase.report(mach).history
    @test map(event -> last(event).measurement[1], history) ≈ results[1:4]

    tm.n=100
    @test_logs (:warn, r"Only 12") fit!(mach, verbosity=0)
    history = MLJBase.report(mach).history
    @test map(event -> last(event).measurement[1], history) ≈ results
end)

end

true

