module TestGrid

using Test
using MLJBase
using MLJTuning
# include("../test/models.jl")
# using .Models
using ..Models
import Random.seed!
seed!(1234)

x1 = rand(100);
x2 = rand(100);
x3 = rand(100)
X = (x1=x1, x2=x2, x3=x3);
y = 2*x1 .+ 5*x2 .- 3*x3 .+ 0.2*rand(100);

mutable struct DummyModel <: Deterministic
    lambda::Float64
    metric::Float64
    kernel::Char
end

dummy_model = DummyModel(4, 9.5, 'k')

mutable struct SuperModel <: Deterministic
    K::Int64
    model1::DummyModel
    model2::DummyModel
end

dummy_model = DummyModel(1.2, 9.5, 'k')
super_model = SuperModel(4, dummy_model, deepcopy(dummy_model))

s = range(super_model, :(model1.kernel), values=['c', 'd'])
r1 = range(super_model, :(model1.lambda), lower=20, upper=31)
r2 = range(super_model, :K, lower=1, upper=11, scale=:log10)

@testset "setup, default_n" begin
    user_range = [r1, (r2, 3), s]

    # with method:
    tuning = Grid(resolution=2, shuffle=false)
    @test MLJTuning.default_n(tuning, user_range) == 12
    models1 =
        params.((MLJTuning.setup(tuning, super_model, user_range, 3)).models)
    tuning = Grid(resolution=2, rng=123)
    @test MLJTuning.default_n(tuning, user_range) == 12
    models1r =
        params.((MLJTuning.setup(tuning, super_model, user_range, 3)).models)

    # by hand:
    m1 = [(K = 1, model1 = (lambda = 20.0, metric = 9.5, kernel = 'c'),
           model2 = (lambda = 1.2, metric = 9.5, kernel = 'k'))
          (K = 1, model1 = (lambda = 31.0, metric = 9.5, kernel = 'c'),
           model2 = (lambda = 1.2, metric = 9.5, kernel = 'k'))
          (K = 3, model1 = (lambda = 20.0, metric = 9.5, kernel = 'c'),
           model2 = (lambda = 1.2, metric = 9.5, kernel = 'k'))
          (K = 3, model1 = (lambda = 31.0, metric = 9.5, kernel = 'c'),
           model2 = (lambda = 1.2, metric = 9.5, kernel = 'k'))
          (K = 11, model1 = (lambda = 20.0, metric = 9.5, kernel = 'c'),
           model2 = (lambda = 1.2, metric = 9.5, kernel = 'k'))
          (K = 11, model1 = (lambda = 31.0, metric = 9.5, kernel = 'c'),
           model2 = (lambda = 1.2, metric = 9.5, kernel = 'k'))
          (K = 1, model1 = (lambda = 20.0, metric = 9.5, kernel = 'd'),
           model2 = (lambda = 1.2, metric = 9.5, kernel = 'k'))
          (K = 1, model1 = (lambda = 31.0, metric = 9.5, kernel = 'd'),
           model2 = (lambda = 1.2, metric = 9.5, kernel = 'k'))
          (K = 3, model1 = (lambda = 20.0, metric = 9.5, kernel = 'd'),
           model2 = (lambda = 1.2, metric = 9.5, kernel = 'k'))
          (K = 3, model1 = (lambda = 31.0, metric = 9.5, kernel = 'd'),
           model2 = (lambda = 1.2, metric = 9.5, kernel = 'k'))
          (K = 11, model1 = (lambda = 20.0, metric = 9.5, kernel = 'd'),
           model2 = (lambda = 1.2, metric = 9.5, kernel = 'k'))
          (K = 11, model1 = (lambda = 31.0, metric = 9.5, kernel = 'd'),
           model2 = (lambda = 1.2, metric = 9.5, kernel = 'k'))]

    @test m1 == models1
    @test m1 != models1r
    @test Set(m1) == Set(models1r)

    # with method:
    tuning = Grid(goal=9, shuffle=false)
    @test MLJTuning.default_n(tuning, user_range) == 8
    models2 =
        params.((MLJTuning.setup(tuning, super_model, user_range, 3)).models)

    # by hand:
    m2 = [(K = 1, model1 = (lambda = 20.0, metric = 9.5, kernel = 'c'),
           model2 = (lambda = 1.2, metric = 9.5, kernel = 'k'))
          (K = 1, model1 = (lambda = 31.0, metric = 9.5, kernel = 'c'),
           model2 = (lambda = 1.2, metric = 9.5, kernel = 'k'))
          (K = 11, model1 = (lambda = 20.0, metric = 9.5, kernel = 'c'),
           model2 = (lambda = 1.2, metric = 9.5, kernel = 'k'))
          (K = 11, model1 = (lambda = 31.0, metric = 9.5, kernel = 'c'),
           model2 = (lambda = 1.2, metric = 9.5, kernel = 'k'))
          (K = 1, model1 = (lambda = 20.0, metric = 9.5, kernel = 'd'),
           model2 = (lambda = 1.2, metric = 9.5, kernel = 'k'))
          (K = 1, model1 = (lambda = 31.0, metric = 9.5, kernel = 'd'),
           model2 = (lambda = 1.2, metric = 9.5, kernel = 'k'))
          (K = 11, model1 = (lambda = 20.0, metric = 9.5, kernel = 'd'),
           model2 = (lambda = 1.2, metric = 9.5, kernel = 'k'))
          (K = 11, model1 = (lambda = 31.0, metric = 9.5, kernel = 'd'),
           model2 = (lambda = 1.2, metric = 9.5, kernel = 'k'))]

    @test models2 == m2

end


@testset "2-parameter tune, with nesting" begin

    sel = FeatureSelector()
    stand = UnivariateStandardizer()
    ridge = FooBarRegressor()
    composite = SimpleDeterministicCompositeModel(transformer=sel, model=ridge)

    features_ = range(composite, :(transformer.features),
                      values=[[:x1], [:x1, :x2], [:x2, :x3], [:x1, :x2, :x3]])
    lambda_ = range(composite, :(model.lambda),
                    lower=1e-6, upper=1e-1, scale=:log10)

    r = [features_, lambda_]

    holdout = Holdout(fraction_train=0.8)
    grid = Grid(resolution=10)

    tuned_model = TunedModel(model=composite, tuning=grid,
                             resampling=holdout, measure=rms,
                             range=r)

    MLJBase.info_dict(tuned_model)

    tuned = machine(tuned_model, X, y)

    fit!(tuned, verbosity=0)
    r = report(tuned)
    @test r.best_report isa NamedTuple{(:machines, :reports)}
    fit!(tuned, verbosity=0)
    rep = report(tuned)
    fp = fitted_params(tuned)
    @test fp.best_fitted_params isa NamedTuple{(:machines, :fitted_params)}
    b = fp.best_model
    @test b isa SimpleDeterministicCompositeModel

    measurements = map(x->x.measurement[1], last.(tuned.report.history))
    # should be all different:
    @test length(unique(measurements)) == length(measurements)

    @test length(b.transformer.features) == 3
    @test abs(b.model.lambda - 0.027825) < 1e-6

    # get the training error of the tuned_model:
    e = rms(y, predict(tuned, X))

    # check this error has same order of magnitude as best measurement
    # during tuning:
    e_training = tuned.report.best_result.measurement[1]
    ratio = e/e_training
    @test ratio < 10 && ratio > 0.1

    # test weights:
    tuned_model.weights = rand(length(y))
    fit!(tuned, verbosity=0)
    @test e_training != tuned.report.best_result.measurement[1]

    # test plotting part of report:
    @test r.plotting.parameter_names ==
        ["transformer.features", "model.lambda"]
    @test r.plotting.parameter_scales == [:none, :log10]
    @test r.plotting.measurements == measurements
    @test size(r.plotting.parameter_values) == (40, 2)

end

@testset "basic tuning with training weights" begin

    seed!(1234)
    N = 100
    X = (x = rand(3N), );
    y = categorical(rand("abc", 3N));
    model = KNNClassifier()
    r = range(model, :K, lower=2, upper=N)
    tuned_model = TunedModel(model=model,
                             tuning=Grid(), measure=BrierScore(),
                             resampling=Holdout(fraction_train=2/3),
                             range=r)

    # no weights:
    tuned = machine(tuned_model, X, y)
    fit!(tuned, verbosity=0)
    best1 = fitted_params(tuned).best_model
    posterior1 = average([predict(tuned, X)...])

    # uniform weights:
    tuned = machine(tuned_model, X, y, fill(1, 3N))
    fit!(tuned, verbosity=0)
    best2 = fitted_params(tuned).best_model
    posterior2 = average([predict(tuned, X)...])

    @test best1 == best2
    @test all([pdf(posterior1, c) ≈ pdf(posterior2, c) for c in levels(y)])

    # skewed weights:
    w = map(y) do η
        if η == 'a'
            return 2
        elseif η == 'b'
            return 4
        else
            return 1
        end
    end
    tuned = machine(tuned_model, X, y, w)
    fit!(tuned, verbosity=0)
    best3 = fitted_params(tuned).best_model
    posterior3 = average([predict(tuned, X)...])

    # different tuning outcome:
    @test best1.K != best3.K

    # "posterior" is skewed appropriately in weighted case:
    @test abs(pdf(posterior3, 'b')/(2*pdf(posterior3, 'a'))  - 1) < 0.15
    @test abs(pdf(posterior3, 'b')/(4*pdf(posterior3, 'c'))  - 1) < 0.15

end


# ## LEARNING CURVE

# @testset "learning curves" begin
#     atom = FooBarRegressor()
#     ensemble = EnsembleModel(atom=atom, n=50, rng=1)
#     mach = machine(ensemble, X, y)
#     r_lambda = range(ensemble, :(atom.lambda),
#                      lower=0.0001, upper=0.1, scale=:log10)
#     curve = MLJ.learning_curve!(mach; range=r_lambda)
#     atom.lambda=0.3
#     r_n = range(ensemble, :n, lower=10, upper=100)
#     curve2 = MLJ.learning_curve!(mach; range=r_n)
#     curve3 = learning_curve(ensemble, X, y; range=r_n)
#     @test curve2.measurements ≈ curve3.measurements
# end

end # module
true
