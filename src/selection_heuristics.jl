abstract type SelectionHeuristic end

## HELPERS

measure_adjusted_weights(weights, measures) =
    if weights isa Nothing
        vcat([signature(measures[1]), ], zeros(length(measures) - 1))
    else
        length(weights) == length(measures) ||
            throw(DimensionMismatch(
                "`OptimizeAggregatedMeasurement` heuristic "*
                "is being applied to a list of measures whose length "*
                "differs from that of the specified `weights`. "))
        signature.(measures) .* weights
    end


## OPTIMIZE AGGREGATED MEASURE

"""
    NaiveSelection(; weights=nothing)

Construct a common selection heuristic for use with `TunedModel` instances
which only considers measurements aggregated over all samples (folds)
in resampling.

For each entry in the tuning history, one defines a penalty equal to
the evaluations of the `measure` specified in the `TunedModel`
instance, aggregated over all samples, and multiplied by `-1` if `measure`
is a `:score`, and `+`` if it is a loss. The heuristic declares as
"best" (optimal) the model whose corresponding entry has the lowest
penalty.

If `measure` is a vector, then the first element is used, unless
per-measure `weights` are explicitly specified. Weights associated
with measures that are neither `:loss` nor `:score` are reset to zero.

"""
struct NaiveSelection <: SelectionHeuristic
    weights::Union{Nothing, Vector{Real}}
end
function NaiveSelection(; weights=nothing)
    if weights isa Vector
        all(x -> x >= 0, weights) ||
            error("`weights` must be non-negative. ")
        end
    return NaiveSelection(weights)
end

function best(heuristic::NaiveSelection, history)
    first_entry = history[1]
    measures = first_entry.measure
    weights = measure_adjusted_weights(heuristic.weights, measures)
    measurements = [weights'*(h.measurement) for h in history]
    best_index = argmin(measurements)
    return history[best_index]
end

MLJTuning.supports_heuristic(::Any, ::NaiveSelection) =
    true
