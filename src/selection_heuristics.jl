abstract type SelectionHeuristic end


## OPTIMIZE AGGREGATED MEASURE

struct OptimizePrimaryAggregatedMeasurement <: SelectionHeuristic end

function best(heuristic::OptimizePrimaryAggregatedMeasurement, history)
    measurements = [h.measurement[1] for h in history]
    measure = first(history).measure[1]
    if orientation(measure) == :score
        measurements = -measurements
    end
    best_index = argmin(measurements)
    return history[best_index]
end

MLJTuning.supports_heuristic(::Any, ::OptimizePrimaryAggregatedMeasurement) =
    true
