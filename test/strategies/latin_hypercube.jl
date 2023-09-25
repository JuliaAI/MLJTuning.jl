module TestLatinHypercube

using Test
using MLJBase
using MLJTuning
using StatisticalMeasures
using LatinHypercubeSampling
import Distributions
import Random
using StableRNGs
using ..Models

const Dist = Distributions

rng = StableRNGs.StableRNG(1234)

x1 = rand(rng, 100);
x2 = rand(rng, 100);
x3 = rand(rng, 100)
X = (x1=x1, x2=x2, x3=x3);
y = 2*x1 .+ 5*x2 .- 3*x3 .+ 0.2*rand(rng, 100);

mutable struct DummyModel <: Deterministic
    lambda::Int
    alpha::Int
    kernel::Char
end

mutable struct SuperModel <: Deterministic
    K::Int64
    model1::DummyModel
    model2::DummyModel
end


MLJBase.fit(::DummyModel, verbosity::Int, X, y) = mean(y), nothing, nothing
MLJBase.predict(::DummyModel, fitresult, Xnew) =
    fill(fitresult, nrows(Xnew))
MLJBase.fit(::SuperModel, verbosity::Int, X, y) = mean(y), nothing, nothing
MLJBase.predict(::SuperModel, fitresult, Xnew) =
    fill(fitresult, nrows(Xnew))

dummy_model = DummyModel(1, 9, 'k')
super_model = SuperModel(4, dummy_model, deepcopy(dummy_model))

@testset "Two ranges with scale" begin
    #ok this works
    model = DummyModel(1,1,'k')
    r1 = range(model, :lambda, lower=1, upper=9);
    r2 = range(model, :alpha, lower=0.4, upper=1.0, scale=:log);
    my_latin = LatinHypercube(gens=2, popsize= 120, rng = rng)
    self_tuning_model = TunedModel(model=model,
                                   tuning=my_latin,
                                   resampling=CV(nfolds=6),
                                   range=[r1, r2],
                                   measure=rms);
end

@testset "Range with infinity" begin
    #ok this works
    model = DummyModel(1, 9, 'k')
    r1 = range(model, :lambda, lower=1, upper=9);
    r2 = range(model, :alpha, lower=0, upper=Inf, origin=2,
               unit=3, scale = :log);
    my_latin = LatinHypercube(gens=4, popsize=100,
                              ntour=2, ptour=0.3,
                              interSampleWeight=0.8, ae_power=2,
                              periodic_ae=false, rng=rng)
    self_tuning_model = TunedModel(model=model,
                                   tuning=my_latin,
                                   resampling=CV(nfolds=6),
                                   range=[r1, r2],
                                   measure=rms);
end

@testset "Full features of latin hypercube" begin
    #ok this works
    model = DummyModel(1, 9, 'k')
    supermodel = SuperModel(4, model, deepcopy(model))

    r1 = range(supermodel, :(model1.lambda), lower=1, upper=9);
    r2 = range(supermodel, :K, lower=0.4, upper=1.5);
    my_latin = LatinHypercube(gens=4, popsize=100,
                              ntour=2, ptour=0.3,
                              interSampleWeight=0.5, ae_power=1.7,
                              periodic_ae=true, rng=rng)

    self_tuning_forest_model = TunedModel(model=supermodel,
                                          tuning=my_latin,
                                          resampling=CV(nfolds=6),
                                          range=[r1, r2],
                                          measure=rms);
end

@testset "_transform" begin
    r_cat = range(Char, :cat, values=['a', 'b', 'c'])
    r_float = range(Float64, :float, lower=1, upper =100, scale=:log10)
    r_int = range(Int32, :int, lower=1, upper=100, scale=:log10)
    @test MLJTuning._transform(r_cat, 2) == 'b'
    @test MLJTuning._transform(r_float, 2.0) == 100.0
    @test MLJTuning._transform(r_int, 2.5) == round(Int32, 10^2.5)
end

@testset "setup" begin
    model = DummyModel(1, 9, 'k')
    r1 = range(model, :lambda, lower=1, upper=9)
    r2 = range(model, :alpha, lower=1, upper=100, scale=:log10);
    my_latin = LatinHypercube(gens=4, popsize=100,
                              ntour=2, ptour=0.3,
                              interSampleWeight=0.5, ae_power=1.7,
                              periodic_ae=true, rng=rng)
    state = MLJTuning.setup(my_latin, model, [r1,r2], 11, 1)
    models = state.models
    @test length(models) == 11
    @test all(models) do model
        model.lambda in 1:9 && model.alpha in 1:100
    end
    @test state.fields == [:lambda, :alpha]
    @test state.parameter_scales == [:linear, :log10]
end

@testset "Scale not a symbol" begin
    #ok this works
    model = DummyModel(1,9,'k')
    r1 = range(model, :lambda, lower=1, upper=9,scale = x->10^x)
    r2 = range(model, :alpha, lower=0.4, upper=1.5);
    @test_throws ArgumentError MLJTuning._create_bounds_and_dims_type(2,[r1,r2])
end

@testset "Return value for ranges" begin
    #ok this works
    model = DummyModel(1,9,'k')
    d = 2
    r1 = range(model, :alpha, lower=1., upper=9.);
    r2 = range(model, :lambda, lower=0.4, upper=1.5);
    bounds, dims = MLJTuning._create_bounds_and_dims_type(d,[r1,r2])
    @test all(bounds[1] .≈ (1.0,9.0))
    @test all(bounds[2] .≈ (0.4,1.5))
    @test all(dims .== [LatinHypercubeSampling.Continuous(),
                        LatinHypercubeSampling.Continuous()])


    r3 = range(model, :lambda, lower=1., upper=9,scale =:log);
    r4 = range(model, :alpha, lower=0.4, upper=1.5);
    bounds, dims = MLJTuning._create_bounds_and_dims_type(d,[r3,r4])
    @test all(bounds[1] .≈ (0.0, log(9)))
    @test all(bounds[2] .≈ (0.4,1.5))
    @test all(dims .== [LatinHypercubeSampling.Continuous(),
                        LatinHypercubeSampling.Continuous()])

    r5 = range(model, :kernel, values=collect("abc"))
    r6 = range(model, :lambda, lower=-Inf, upper=+Inf,
               origin = 0., unit = 3.)
    bounds, dims = MLJTuning._create_bounds_and_dims_type(d,[r5,r6])
    @test all(bounds[1] .≈ (1,3))
    @test all(bounds[2] .≈ (-3.0, 3.0))
    @test all(dims .== [LatinHypercubeSampling.Categorical(3,1.0),
                       LatinHypercubeSampling.Continuous()])

    r7 = range(model, :lambda, lower=0., upper=1.0)
    r8 = range(model, :alpha, lower = -Inf, upper = 15.,
               origin = 4., unit = 10., scale =:linear)
    bounds, dims = MLJTuning._create_bounds_and_dims_type(d,[r7,r8])
    @test all(bounds[1] .≈ (0.0, 1.0))
    @test all(bounds[2] .≈ (-5.,15.0))
    @test all(dims .== [LatinHypercubeSampling.Continuous(),
                        LatinHypercubeSampling.Continuous()])

    r9 = range(model, :alpha, lower=-Inf, upper=+Inf,
               origin = 5.0, unit = 1.0, scale = :log2)
    r10 = range(model, :lambda, lower = 5.0, upper = +Inf,
                origin = 10., unit = 10., scale =:linear)
    bounds, dims = MLJTuning._create_bounds_and_dims_type(d,[r9,r10])
    @test all(bounds[1] .≈ (2.0, log(2, 6)))
    @test all(bounds[2] .≈ (5.0, 25.0))
    @test all(dims .== [LatinHypercubeSampling.Continuous(),
                        LatinHypercubeSampling.Continuous()])

end

@testset "integration test 1 - dummy model" begin

    model = DummyModel(1, 9, 'k')
    supermodel = SuperModel(4, model, deepcopy(model))

    r2 = range(supermodel, :K, lower=10, upper=100, scale=:log10)
    r3 = range(supermodel, :(model2.kernel), values = collect("abcdefghijk"))
    tuning = LatinHypercube()
    r = [r2, r3]
    holdout = Holdout(fraction_train=0.8)
    tuned_model = TunedModel(n =5,
                             model=supermodel,
                             tuning=tuning,
                             resampling=holdout,
                             range=r,
                             measure=rms);

    tuned = machine(tuned_model, X, y)

    fit!(tuned, verbosity=0)
    rep = MLJBase.report(tuned)
    models = map(first, rep.history)
    @test length(models) == 5
    @test all(models) do model
        model.K in 1:100 && model.model2.kernel in collect("abcdefghijk")
    end

end

@testset "integration test 2 - compare with grid search in real case" begin

    X, y = @load_boston();
    model = DecisionTreeRegressor()

    r = range(model, :min_samples_split, lower=2, upper=20)

    # latin:
    tuning = LatinHypercube(rng=rng)
    tuned_model = TunedModel(model=model,
                             tuning=tuning,
                             resampling=Holdout(),
                             range=r,
                             measure=rms,
                             n=100);
    tuned = machine(tuned_model, X, y)
    fit!(tuned, verbosity=0)
    best_latin = fitted_params(tuned).best_model.min_samples_split
    best_rms_latin = report(tuned).best_history_entry.measurement

    # grid:
    tuning = Grid(goal=100, rng=rng)
    tuned_model = TunedModel(model=model,
                             tuning=tuning,
                             resampling=Holdout(),
                             range=r,
                             measure=rms,
                             n=100);
    tuned = machine(tuned_model, X, y)
    fit!(tuned, verbosity=0)
    best_grid= fitted_params(tuned).best_model.min_samples_split
    best_rms_grid = report(tuned).best_history_entry.measurement

    @test best_latin == best_grid
    @test best_rms_latin ≈ best_rms_grid

end

end # module

true
