module TestRandomSearch

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

@testset "Constructor" begin
    @test_throws Exception RandomSearch(bounded=Dist.Uniform(1,2))
    @test_throws Exception RandomSearch(positive_unbounded=Dist.Poisson(1))
    @test_throws Exception RandomSearch(bounded=Dist.Uniform(1,2))
end

@testset "setup" begin
    rng = StableRNGs.StableRNG(1234)
    user_range = [r0, (r1, Dist.SymTriangularDist), r2]
    tuning = RandomSearch(positive_unbounded=Dist.Gamma, rng=rng)

    @test MLJTuning.default_n(tuning, user_range) == MLJTuning.DEFAULT_N

    p0, p1, p2 = MLJTuning.setup(tuning, super_model, user_range, 3)
    @test first.([p0, p1, p2]) == [:(model1.kernel), :(model1.lambda), :K]

    s0, s1, s2 = last.([p0, p1, p2])
    @test s0.distribution == Dist.Categorical(0.5, 0.5)
    @test s1.distribution == Dist.SymTriangularDist(2,1)
    γ = s2.distribution
    @test mean(γ) == 2
    @test std(γ) == 3
end

@testset "models!" begin
    rng = StableRNGs.StableRNG(1234)
    N = 10000
    model = DummyModel(1, 1, 'k')
    r1 = range(model, :lambda, lower=0, upper=1)
    r2 = range(model, :alpha, lower=-1, upper=1)
    user_range = [r1, r2]
    tuning = RandomSearch(rng=rng)
    tuned_model = TunedModel(model=model,
                             tuning=tuning,
                             n=N,
                             range=user_range,
                             measures=[rms,mae])
    state = MLJTuning.setup(tuning, model, user_range, 3)
    my_models = MLJTuning.models!(tuning,
                               model,
                               nothing, # history
                               state,
                               N,       # n_remaining
                               0)

    # check the samples of each hyperparam have expected distritution:
    lambdas = map(m -> m.lambda, my_models)
    alphas = map(m -> m.alpha, my_models)
    a, b = values(Dist.countmap(lambdas))
    @test abs(a/b - 1) < 0.06
    dict = Dist.countmap(alphas)
    a, b, c = dict[-1], dict[0], dict[1]
    @test abs(b/a - 2) < 0.06
    @test abs(b/c - 2) < 0.06
end

@testset "tuned model using random search and its report" begin
    rng = StableRNGs.StableRNG(1234)
    N = 4
    model = DummyModel(1, 1, 'k')
    r1 = range(model, :lambda, lower=0, upper=1)
    r2 = range(model, :alpha, lower=-1, upper=1)
    user_range = [r1, r2]
    tuning = RandomSearch(rng=rng)
    tuned_model = TunedModel(model=model,
                             tuning=tuning,
                             n=N,
                             resampling=Holdout(fraction_train=0.5),
                             range=user_range,
                             measures=[rms,mae])
    mach = machine(tuned_model, X, y)
    fit!(mach, verbosity=0)

    # model predicts mean of training target, so:
    train, test = partition(eachindex(y), 0.5)
    μ = mean(y[train])
    error = mean((y[test] .- μ).^2) |> sqrt

    r = report(mach)
    @test r.plotting.parameter_names ==
        ["lambda", "alpha"]
    @test r.plotting.parameter_scales == [:linear, :linear]
    @test r.plotting.measurements ≈ fill(error, N)
    @test size(r.plotting.parameter_values) == (N, 2)
end

struct ConstantSampler
    c
end
Base.rand(rng::Random.AbstractRNG, s::ConstantSampler) = s.c

@testset "multiple samplers for single field" begin
    rng = StableRNGs.StableRNG(1234)
    N = 1000
    model = DummyModel(1, 1, 'k')
    r = range(model, :alpha, lower=-1, upper=1)
    user_range = [(:lambda, ConstantSampler(0)),
             r,
             (:lambda, ConstantSampler(1))]
    tuning = RandomSearch(rng=rng)
    tuned_model = TunedModel(model=model,
                             tuning=tuning,
                             n=N,
                             range=user_range,
                             measures=[rms,mae])
    mach = fit!(machine(tuned_model, X, y))
    my_models = first.(report(mach).history);
    lambdas = map(m -> m.lambda, my_models);
    a, b = values(Dist.countmap(lambdas))
    @test abs(a/b -1) < 0.06
    @test a + b == N
end

end # module
true
