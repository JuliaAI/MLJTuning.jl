module TestLatinHypercube

using Test
using MLJBase
using MLJTuning
import Distributions
import Random
using StableRNGs

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
    fill(fitresult, schema(Xnew).nrows)

dummy_model = DummyModel(1, 9, 'k')
super_model = SuperModel(4, dummy_model, deepcopy(dummy_model))

r0 = range(super_model, :(model1.kernel), values=['c', 'd'])
r1 = range(super_model, :(model1.lambda), lower=1, upper=3)
r2 = range(super_model, :K, lower=0, upper=Inf, origin=2, unit=3)

@testset "Two ranges with scale" begin

    model = DummyModel(1,1,'k')
    r1 = range(model, :lambda, lower=1, upper=9);
    r2 = range(model, :alpha, lower=0.4, upper=1.0, scale=:log);
    my_latin = LatinHypercube(nGenerations=2,popSize= 120, rng = rng)
    self_tuning_forest_model = TunedModel(model=forest_model,
                                          tuning=my_latin,
                                          resampling=CV(nfolds=6),
                                          range=[r1, r2],
                                          measure=rms);
end

#=
@testset "Range with infinity" begin
    rng = StableRNGs.StableRNG(1234)
    x1 = rand(rng, 100);
    x2 = rand(rng, 100);
    x3 = rand(rng, 100)
    X = (x1=x1, x2=x2, x3=x3);
    y = 2*x1 .+ 5*x2 .- 3*x3 .+ 0.2*rand(rng, 100);

    tree_model = DecisionTreeRegressor()
    forest_model = EnsembleModel(atom=tree_model)

    r1 = range(forest_model, :(atom.n_subfeatures), lower=1, upper=9);
    r2 = range(forest_model, :bagging_fraction,
               lower=0, upper=Inf, origin=2, unit=3, scale = :log);
    my_latin = LatinHypercube(nGenerations=2,popSize= 120, rng = rng)
    self_tuning_forest_model = TunedModel(model=forest_model,
                                          tuning=my_latin,
                                          resampling=CV(nfolds=6),
                                          range=[r1, r2],
                                          measure=rms);
end

@testset "Full features of latin hypercube" begin
    rng = StableRNGs.StableRNG(1234)
    x1 = rand(rng, 100);
    x2 = rand(rng, 100);
    x3 = rand(rng, 100)
    X = (x1=x1, x2=x2, x3=x3);
    y = 2*x1 .+ 5*x2 .- 3*x3 .+ 0.2*rand(rng, 100);

    tree_model = DecisionTreeRegressor()
    forest_model = EnsembleModel(atom=tree_model)

    r1 = range(forest_model, :(atom.n_subfeatures), lower=1, upper=9);
    r2 = range(forest_model, :bagging_fraction, lower=0.4, upper=1.5);
    my_latin = LatinHypercube(nGenerations=4,popSize=100, nTournament = 2,
                              pTournament=0.3, interSampleWeight = 1.5,
                              ae_power = 1.7, periodic_ae = true, rng = rng)

    self_tuning_forest_model = TunedModel(model=forest_model,
                                          tuning=my_latin,
                                          resampling=CV(nfolds=6),
                                          range=[r1, r2],
                                          measure=rms);
end

@testset "setup" begin
    rng = StableRNGs.StableRNG(1234)
    tree_model = DecisionTreeRegressor()
    forest_model = EnsembleModel(atom=tree_model)
    r1 = range(forest_model, :(atom.n_subfeatures), lower=1, upper=9)
    r2 = range(forest_model, :bagging_fraction, lower=0.4, upper=1.5);
    my_latin = LatinHypercube(nGenerations=2,popSize= 120, rng = rng)
    MLJTuning.setup(my_latin, forest_model, [r1,r2], 1)
end

@testset "Scale not a symbol" begin
    rng = StableRNGs.StableRNG(1234)
    tree_model = DecisionTreeRegressor()
    forest_model = EnsembleModel(atom=tree_model)
    r1 = range(forest_model, :(atom.n_subfeatures), lower=1, upper=9,
               scale = x->10^x)
    r2 = range(forest_model, :bagging_fraction, lower=0.4, upper=1.5);
    my_latin = LatinHypercube(nGenerations=2,popSize= 120, rng = rng)
    @test_throws ErrorException setup(tuning::LatinHypercube, model=forest_model,
                                      r=[r1,r2], verbosity)
end

@testset "Return value for ranges" begin
    tree_model = DecisionTreeRegressor()
    forest_model = EnsembleModel(atom=tree_model)
    d = 2

    r1 = range(forest_model, :(atom.n_subfeatures), lower=1., upper=9.);
    r2 = range(forest_model, :bagging_fraction, lower=0.4, upper=1.5);
    bounds, dims = MLJTuning._create_bounds_and_dims(d,[r1,r2])
    @test all(bounds .≈ [(1,9),(0.4,1.5)])
    @test all(dims .≈ [Continuous(),Continuous()])

    r3 = range(forest_model, :(atom.n_subfeatures), lower=1., upper=9,
               scale =:log);
    r4 = range(forest_model, :bagging_fraction, lower=0.4, upper=1.5);
    bounds, dims = MLJTuning._create_bounds_and_dims(d,[r3,r4])
    @test all(bounds .≈ [(0.0, 2.1972245773362196),(0.4,1.5)])
    @test all(dims .≈ [Continuous(),Continuous()])

    r5 = range(Char, :letter, values=collect("abc"))
    r6 = range(forest_model, :(atom.n_subfeatures), lower=-Inf, upper=+Inf,
               origin = 0., unit = 3.)
    bounds, dims = MLJTuning._create_bounds_and_dims(d,[r5,r6])
    @test all(bounds .≈[(0, 3), (-3.0, 3.0)])
    @test all(dims .≈ [Categorical(3,1.0),Continuous()])

    r7 = range(forest_model, :(atom.n_subfeatures), lower=0., upper=1.0)
    r8 = range(forest_model, :bagging_fraction, lower = -Inf, upper = 15.,
               origin = 4., unit = 10., scale =:linear)
    bounds, dims = MLJTuning._create_bounds_and_dims(d,[r7,r8])
    @test all(bounds .≈ [(0.0, 1.0),(-5.,15.0)])
    @test all(dims .≈ [Continuous(),Continuous()])


    r9 = range(forest_model, :(atom.n_subfeatures), lower=-Inf, upper=+Inf,
               origin = 5.0, unit = 1.0, scale = :log2)
    r10 = range(forest_model, :bagging_fraction, lower = 5.0, upper = +Inf,
                origin = 10., unit = 10., scale =:linear)
    bounds, dims = MLJTuning._create_bounds_and_dims(d,[r9,r10])
    @test all(bounds .≈ [(2.0, 2.584962500721156),(5.0, 25.0)])
    @test all(dims .≈ [Continuous(),Continuous()])
end
=#
end
