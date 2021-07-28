good = KNNClassifier(K=2)
bad = KNNClassifier(K=10)
ugly = ConstantClassifier()

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

end

true
