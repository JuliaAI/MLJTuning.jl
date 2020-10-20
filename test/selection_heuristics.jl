using .Models

measures = [accuracy, confmat, misclassification_rate]
@test MLJTuning.measure_adjusted_weights([2, 3, 4], measures) == [-2, 0, 4]
@test MLJTuning.measure_adjusted_weights(nothing, measures) == [-1, 0, 0]
@test_throws(DimensionMismatch,
             MLJTuning.measure_adjusted_weights([2, 3], measures))

@testset "losses/scores get minimized/maximimized" begin
    bad_model = KNNClassifier(K=100)
    good_model = KNNClassifier(K=5)

    am = [accuracy, misclassification_rate]
    ma = [misclassification_rate, accuracy]

    # scores when `weights=nothing`
    history = [(model=bad_model, measure=am, measurement=[0, 1]),
               (model=good_model, measure=am, measurement=[1, 0])]
    @test MLJTuning.best(NaiveSelection(), history).model == good_model

    # losses when `weights=nothing`
    history = [(model=bad_model, measure=ma, measurement=[1, 0]),
               (model=good_model, measure=ma, measurement=[0, 1])]
    @test MLJTuning.best(NaiveSelection(), history).model == good_model

    # mixed case favouring the score:
    weights = [2, 1]
    history = [(model=bad_model, measure=am, measurement=[0, 0]),
               (model=good_model, measure=am, measurement=[1, 1])]
    heuristic = NaiveSelection(weights=weights)
    @test MLJTuning.best(heuristic, history).model == good_model

    # mixed case favouring the loss:
    weights = [1, 2]
    history = [(model=bad_model, measure=am, measurement=[1, 1]),
               (model=good_model, measure=am, measurement=[0, 0])]
    heuristic = NaiveSelection(weights=weights)
    @test MLJTuning.best(heuristic, history).model == good_model
end
