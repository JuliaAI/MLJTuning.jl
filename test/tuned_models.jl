using Distributed

using Test

@everywhere begin
    using MLJBase
    using MLJTuning
    using ..Models
    import ComputationalResources: CPU1, CPUProcesses, CPUThreads
end

using Random
Random.seed!(1234*myid())
using .TestUtilities

N = 30
x1 = rand(N);
x2 = rand(N);
x3 = rand(N);
X = (x1=x1, x2=x2, x3=x3);
y = 2*x1 .+ 5*x2 .- 3*x3 .+ 0.4*rand(N);

m(K) = KNNRegressor(K=K)
r = [m(K) for K in 13:-1:2]

# TODO: replace the above with the line below and post an issue on
# the failure (a bug in Distributed, I reckon):
# r = (m(K) for K in 13:-1:2)

@testset "constructor" begin
    @test_throws(MLJTuning.ERR_SPECIFY_RANGE,
                 TunedModel(model=first(r), tuning=Grid(), measure=rms))
    @test_throws(MLJTuning.ERR_SPECIFY_RANGE,
                 TunedModel(model=first(r), measure=rms))
    @test_throws(MLJTuning.ERR_BOTH_DISALLOWED,
                 TunedModel(model=first(r),
                            models=r, tuning=Explicit(), measure=rms))
    tm = TunedModel(models=r, tuning=Explicit(), measure=rms)
    @test tm.tuning isa Explicit && tm.range ==r && tm.model == first(r)
    @test input_scitype(tm) == Unknown
    @test TunedModel(models=r, measure=rms) == tm
    @test_logs (:info, r"No measure") @test TunedModel(models=r) == tm

    @test_throws(MLJTuning.ERR_SPECIFY_MODEL,
                 TunedModel(range=r, measure=rms))
    @test_throws(MLJTuning.ERR_MODEL_TYPE,
                 TunedModel(model=42, tuning=Grid(),
                            range=r, measure=rms))
    @test_logs (:info, MLJTuning.INFO_MODEL_IGNORED) tm =
        TunedModel(model=42, tuning=Explicit(), range=r, measure=rms)
    @test_logs (:info, r"No measure") tm =
        TunedModel(model=first(r), range=r)
    @test_throws(MLJTuning.ERR_SPECIFY_RANGE_OR_MODELS,
                TunedModel(tuning=Explicit(), measure=rms))
    @test_throws(MLJTuning.ERR_NEED_EXPLICIT,
                 TunedModel(models=r, tuning=Grid()))
    tm = TunedModel(model=first(r), range=r, measure=rms)
    @test tm.tuning isa Grid
    @test input_scitype(tm) == Table(Continuous)
end

results = [(evaluate(model, X, y,
                     resampling=CV(nfolds=2),
                     measure=rms,
                     verbosity=0,)).measurement[1] for model in r]

@testset "measure compatibility check" begin
    tm = TunedModel(model=first(r), tuning=Explicit(),
                    range=r, resampling=CV(nfolds=2),
                    measures=cross_entropy)
    @test_logs((:error, r"Problem"),
               (:info, r""),
               (:info, r""),
               @test_throws ArgumentError fit(tm, 0, X, y))
end

@testset_accelerated "basic fit (CPU1)" accel begin
    printstyled("\n Testing progressmeter basic fit with $(accel) and CPU1 resampling \n", color=:bold)
    best_index = argmin(results)
    tm = TunedModel(model=first(r), tuning=Explicit(),
                    range=r, resampling=CV(nfolds=2),
                    measures=[rms, l1], acceleration=accel)
    verbosity = accel isa CPU1 ? 2 : 1
    fitresult, meta_state, _report = fit(tm, verbosity, X, y);
    history, _, state = meta_state;
    results2 = map(event -> event.measurement[1], history)
    @test results2 ≈ results
    @test fitresult.model == collect(r)[best_index]
    @test _report.best_model == collect(r)[best_index]
    @test _report.history[5] == MLJTuning.delete(history[5], :metadata)

    # training_losses:
    losses = training_losses(tm, _report)
    @test all(eachindex(losses)) do i
        minimum(results[1:i]) == losses[i]
    end
    @test MLJBase.iteration_parameter(tm) == :n

end

@static if VERSION >= v"1.3.0-DEV.573"
@testset_accelerated "Basic fit (CPUThreads)" accel begin
    printstyled("\n Testing progressmeter basic fit with $(accel) and CPUThreads resampling \n", color=:bold)
    tm = TunedModel(model=first(r), tuning=Explicit(),
                    range=r, resampling=CV(nfolds=2),
                    measures=[rms, l1], acceleration= CPUThreads(),
                    acceleration_resampling=accel)
    fitresult, meta_state, report = fit(tm, 1, X, y);
    history, _, state = meta_state;
    results3 = map(event -> event.measurement[1], history)
    @test results3 ≈ results
end
end
@testset_accelerated "Basic fit (CPUProcesses)" accel begin
    printstyled("\n Testing progressmeter basic fit with $(accel) and CPUProcesses resampling \n", color=:bold)
    best_index = argmin(results)
    tm = TunedModel(model=first(r), tuning=Explicit(),
                    range=r, resampling=CV(nfolds=2),
                    measures=[rms, l1], acceleration=CPUProcesses(),
                    acceleration_resampling=accel)
    fitresult, meta_state, report = fit(tm, 1, X, y);
    history, _, state = meta_state;
    results4 = map(event -> event.measurement[1], history)
    @test results4 ≈ results
end

@testset_accelerated("under/over supply of models", accel, begin
    tm = TunedModel(model=first(r), tuning=Explicit(),
                    range=r, measures=[rms, l1],
                    acceleration=accel,
                    resampling=CV(nfolds=2),
                    n=4)
    mach = machine(tm, X, y)
    fit!(mach, verbosity=0)
    history = MLJBase.report(mach).history
    @test map(event -> event.measurement[1], history) ≈ results[1:4]

    tm.n += 2
    @test_logs((:info, r"^Updating"),
               (:info, r"^Attempting to add 2.*to 6"),
               fit!(mach, verbosity=1))
    history = MLJBase.report(mach).history
    @test map(event -> event.measurement[1], history) ≈ results[1:6]

    tm.n=100
    @test_logs (:info, r"Only 12") fit!(mach, verbosity=0)
    history = MLJBase.report(mach).history
    @test map(event -> event.measurement[1], history) ≈ results
end)

@everywhere begin

    # variation of the Explicit strategy that annotates the models
    # with metadata
    mutable struct MockExplicit <: MLJTuning.TuningStrategy end

    annotate(model) = (model, params(model)[1])

    _length(x) = length(x)
    _length(::Nothing) = 0
    MLJTuning.models(tuning::MockExplicit,
                     model,
                     history,
                     state,
                     n_remaining,
                     verbosity) =
                         annotate.(state)[_length(history) + 1:end], state

    function default_n(tuning::Explicit, range)
        try
            length(range)
        catch MethodError
            DEFAULT_N
        end

    end

end

@testset_accelerated("passing of model metadata", accel,
                  begin
                     tm = TunedModel(model=first(r), tuning=MockExplicit(),
                                     range=r, resampling=CV(nfolds=2),
                                     measures=[rms, l1], acceleration=accel)
                     fitresult, meta_state, report = fit(tm, 0, X, y);
                     history, _, state = meta_state;
                     @test all(history) do event
                         event.metadata == event.model.K
                     end
                     end)



@testset "data caching" begin
    X = (x1= ones(5),);
    y = coerce(collect("abcaa"), Multiclass);

    m(b) = ConstantClassifier(testing=true, bogus=b)
    r = [m(b) for b in 1:5]

    tuned_model = TunedModel(model=first(r), tuning=Explicit(),
                             range=r, resampling=Holdout(fraction_train=0.8),
                             measure=log_loss, cache=true)

    # There is reformatting and resampling of X and y for training of
    # first model, and then resampling on the prediction side only
    # thereafter. Then, finally, for training on all supplied data, there
    # is reformatting and resampling again:
    @test_logs(
        (:info, "reformatting X, y"), # fit 1
        (:info, "resampling X, y"),   # fit 1
        (:info, "resampling X"),      # predict 1
        (:info, "resampling X"),      # predict 2
        (:info, "resampling X"),      # predict 3
        (:info, "resampling X"),      # predict 4
        (:info, "resampling X"),      # predict 5
        (:info, "reformatting X, y"), # fit on all data
        (:info, "resampling X, y"),   # fit one all data
        fit(tuned_model, 0, X, y)
    );

    # Otherwise, resampling and reformatting happen for every model
    # trained, and we get a reformat on the prediction side every
    # time:
    tuned_model.cache = false
    @test_logs(
        (:info, "reformatting X, y"), # fit 1
        (:info, "resampling X, y"),   # fit 1
        (:info, "reformatting X"),    # predict 1
        (:info, "reformatting X, y"), # fit 2
        (:info, "resampling X, y"),   # fit 2
        (:info, "reformatting X"),    # predict 2
        (:info, "reformatting X, y"), # fit 3
        (:info, "resampling X, y"),   # fit 3
        (:info, "reformatting X"),    # predict 3
        (:info, "reformatting X, y"), # fit 4
        (:info, "resampling X, y"),   # fit 4
        (:info, "reformatting X"),    # predict 4
        (:info, "reformatting X, y"), # fit 5
        (:info, "resampling X, y"),   # fit 5
        (:info, "reformatting X"),    # predict 5
        (:info, "reformatting X, y"), # fit on all data
        (:info, "resampling X, y"),   # fit on all data
        fit(tuned_model, 0, X, y)
    );
end

@testset "issue #128" begin
    X, y = make_regression(10, 2)
    dtc = DecisionTreeRegressor()
    r   = range(dtc, :max_depth, lower=1, upper=50);

    tmodel = TunedModel(model=dtc, ranges=[r, ],
                        tuning=Grid(resolution=50),
                        measure=mae,
                        n=48);
    mach = machine(tmodel, X, y)
    fit!(mach, verbosity=0);

    @test length(report(mach).history) == 48

    tmodel.n = 49
    fit!(mach, verbosity=0);

    @test length(report(mach).history) == 49
end

true
