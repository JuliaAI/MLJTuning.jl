good = KNNClassifier(K=2)
bad = KNNClassifier(K=10)
ugly = ConstantClassifier()
evil = DeterministicConstantClassifier()

r = [good, bad, ugly]

rng = StableRNGs.StableRNG(123)
X, y = make_blobs(rng=rng)

@testset_accelerated "integration" accel begin

    # evaluate the three models separately:
    resampling = Holdout(fraction_train=0.6)
    scores = map(r) do model
        e = evaluate(model, X, y, resampling=resampling, measure=LogLoss())
        return e.measurement[1]
    end
    @test scores == sort(scores)

    # find the scores using `Explicit` tuning:
    tmodel = TunedModel(models=r,
                        resampling=resampling,
                        measure=LogLoss(),
                        acceleration=accel)
    mach = machine(tmodel, X, y)
    fit!(mach, verbosity=0)

    history = report(mach).history
    _models = first.(history)
    _scores = map(history) do entry
        entry.measurement[1]
    end
    ms = sort(zip(_models, _scores) |> collect, by=last)

    # check ordering is best to worst again:
    @test first.(ms) == r

    # check scores are the same:
    @test last.(ms) ≈ scores

    # fail with ArgumentError when model types are wrong (e.g., are not instantiated).
    # this used to throw a very confusing MethodError.
    dcc = DeterministicConstantClassifier
    @test_throws ArgumentError TunedModel(; models=[dcc, dcc])
end

r = [good, bad, evil, ugly]
r = [good, bad, ugly]
r= [evil,]

@testset "inconsistent prediction types" begin
    # case where different predictions types is actually okay (but still
    # a warning is issued):
    tmodel = TunedModel(
       models=r,
        resampling = Holdout(),
        measure=accuracy,
    )
    @test_logs(
    (:warn, MLJTuning.WARN_INCONSISTENT_PREDICTION_TYPE),
    MLJBase.fit(tmodel, 0, X, y),
    );

    # verbosity = -1 suppresses the warning:
    @test_logs(
    MLJBase.fit(tmodel, -1, X, y),
    );

    # case where there really is a problem with different prediction types:
    tmodel = TunedModel(
       models=r,
        resampling = Holdout(),
        measure=log_loss,
    )
    @test_logs(
        (:warn, MLJTuning.WARN_INCONSISTENT_PREDICTION_TYPE),
        (:error,),
        (:info,),
        (:info,),
        @test_throws(
            ArgumentError, # indicates the problem is with incompatible measure
            MLJBase.fit(tmodel, 0, X, y),
        )
    )
end

true
