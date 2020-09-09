measures = [accuracy, confmat, misclassification_rate]
@test MLJTuning.signature.(measures) == [-1, 0, 1]
@test MLJTuning.measure_adjusted_weights([2, 3, 4], measures) == [-2, 0, 4]
@test MLJTuning.measure_adjusted_weights(nothing, measures) == [-1, 0, 0]
@test_throws(DimensionMismatch,
             MLJTuning.measure_adjusted_weights([2, 3], measures))
