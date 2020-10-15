module TestLatinHyoercube

using Test
using MLJBase
using MLJTuning
using ..Models
using StableRNGs

rng=StableRNGs.StableRNG(1234)

@testset "Two ranges with scale" begin
x1 = rand(rng, 100);
x2 = rand(rng, 100);
x3 = rand(rng, 100)
X = (x1=x1, x2=x2, x3=x3);
y = 2*x1 .+ 5*x2 .- 3*x3 .+ 0.2*rand(rng, 100);

tree_model = DecisionTreeRegressor()
forest_model = EnsembleMode(atom=tree_model)

r1 = range(forest_model, :(atom.n_subfeatures), lower=1, upper=9);
r2 = range(forest_model, :bagging_fraction, lower=0.4, upper=1.0, scale=:log);
my_latin = LatinHypercube(nGenerations=2,popSize= 120)
self_tuning_forest_model = TunedModel(model=forest_model,
                                             tuning=my_latin,
                                             resampling=CV(nfolds=6),
                                             range=[r1, r2],
                                             measure=rms);
end

@testset "Range with infinity"
x1 = rand(rng, 100);
x2 = rand(rng, 100);
x3 = rand(rng, 100)
X = (x1=x1, x2=x2, x3=x3);
y = 2*x1 .+ 5*x2 .- 3*x3 .+ 0.2*rand(rng, 100);

tree_model = DecisionTreeRegressor()
forest_model = EnsembleMode(atom=tree_model)

r1 = range(forest_model, :(atom.n_subfeatures), lower=1, upper=9);
r2 = range(forest_model, :bagging_fraction,
           lower=0, upper=Inf, origin=2, unit=3, scale = :log);
my_latin = LatinHypercube(nGenerations=2,popSize= 120)
self_tuning_forest_model = TunedModel(model=forest_model,
                                             tuning=my_latin,
                                             resampling=CV(nfolds=6),
                                             range=[r1, r2],
                                             measure=rms);
end


@testset "Full features of latin hypercube" begin
x1 = rand(rng, 100);
x2 = rand(rng, 100);
x3 = rand(rng, 100)
X = (x1=x1, x2=x2, x3=x3);
y = 2*x1 .+ 5*x2 .- 3*x3 .+ 0.2*rand(rng, 100);

tree_model = DecisionTreeRegressor()
forest_model = EnsembleMode(atom=tree_model)

r1 = range(forest_model, :(atom.n_subfeatures), lower=1, upper=9);
r2 = range(forest_model, :bagging_fraction, lower=0.4, upper=1.5);
my_latin = LatinHypercube(nGenerations=4,popSize=100, nTournament = 2,
                          pTournment=0.3, interSampleWeight = 1.5,
                          ae_power = 1.7, periodic_ae = true)

self_tuning_forest_model = TunedModel(model=forest_model,
                                             tuning=my_latin,
                                             resampling=CV(nfolds=6),
                                             range=[r1, r2],
                                             measure=rms);
end

@testset "setup" begin
    tree_model = DecisionTreeRegressor()
    forest_model = EnsembleMode(atom=tree_model)
    r1 = range(forest_model, :(atom.n_subfeatures), lower=1, upper=9)
    r2 = range(forest_model, :bagging_fraction, lower=0.4, upper=1.5);
    my_latin = LatinHypercube(nGenerations=2,popSize= 120)
    setup(tuning::LatinHypercube, model=forest_model,r=[r1,r2], verbosity)
end

@testset "Scale not a symbol" begin
tree_model = DecisionTreeRegressor()
forest_model = EnsembleMode(atom=tree_model)
r1 = range(forest_model, :(atom.n_subfeatures), lower=1, upper=9,
           scale = x->10^x)
r2 = range(forest_model, :bagging_fraction, lower=0.4, upper=1.5);
my_latin = LatinHypercube(nGenerations=2,popSize= 120)
@test_throws ErrorException setup(tuning::LatinHypercube, model=forest_model,
                                  r=[r1,r2], verbosity)
end
