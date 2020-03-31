module TestRandomSearch

using Test
using MLJBase
using MLJTuning
import Distributions
import Random
import Random.seed!
seed!(1234)

const Dist = Distributions

x1 = rand(100);
x2 = rand(100);
x3 = rand(100)
X = (x1=x1, x2=x2, x3=x3);
y = 2*x1 .+ 5*x2 .- 3*x3 .+ 0.2*rand(100);

mutable struct DummyModel <: Deterministic
    lambda::Int
    metric::Int
    kernel::Char
end

mutable struct SuperModel <: Deterministic
    K::Int64
    model1::DummyModel
    model2::DummyModel
end

MLJBase.fit(::DummyModel, verbosity::Int, X, y) = std(y), nothing, nothing
MLJBase.predict(::DummyModel, fitresult, Xnew)  = fitresult

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
    user_range = [r0, (r1, Dist.SymTriangularDist), r2]
    tuning = RandomSearch(positive_unbounded=Dist.Gamma, rng=123)

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
    N = 10000
    model = DummyModel(1, 1, 'k')
    r1 = range(model, :lambda, lower=0, upper=1)
    r2 = range(model, :metric, lower=-1, upper=1)
    user_range = [r1, r2]
    tuning = RandomSearch(rng=1)
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
    metrics = map(m -> m.metric, my_models)
    a, b = values(Dist.countmap(lambdas))
    @test abs(a/b - 1) < 0.06
    dict = Dist.countmap(metrics)
    a, b, c = dict[-1], dict[0], dict[1]
    @test abs(b/a - 2) < 0.06
    @test abs(b/c - 2) < 0.06
end

end # module
true
