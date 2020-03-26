## TYPES TO BE SUBTYPED

abstract type TuningStrategy <: MLJBase.MLJType end
MLJBase.show_as_constructed(::Type{<:TuningStrategy}) = true


## METHODS TO BE IMPLEMENTED

# for initialization of state (compulsory)
setup(tuning::TuningStrategy, model, range, verbosity) = range

# for building each element of the history:
result(tuning::TuningStrategy, history, state, e, metadata) =
    (measure=e.measure, measurement=e.measurement)

# for generating batches of new models and updating the state (but not
# history):
function models! end

# for extracting the optimal model (and its performance) from the
# history:
function best(tuning::TuningStrategy, history)
   measurements = [h[2].measurement[1] for h in history]
   measure = first(history)[2].measure[1]
   if orientation(measure) == :score
       measurements = -measurements
   end
   best_index = argmin(measurements)
   return history[best_index]
end

# for selecting what to report to the user apart from the optimal
# model:
tuning_report(tuning::TuningStrategy, history, state) = (history=history,)

# for declaring the default number of models to evaluate:
default_n(tuning::TuningStrategy, range) = DEFAULT_N


## CONVENIENCE METHODS

"""
    MLJTuning.isrecorded(model, history)
    MLJTuning.isrecorded(model, history, exceptions::Symbol...)

Test if `history` has an entry for some model `m` sharing the same
hyperparameter values as `model`, with the possible exception of fields
specified in `exceptions`.

More precisely, the requirement is that
`MLJModelInterface.is_same_except(m, model, exceptions...)` be true.

"""
isrecorded(model::MLJBase.Model, ::Nothing) = false
function isrecorded(model::MLJBase.Model, history)::Bool
    for (metamodel, _) in history
        MLJModelInterface.is_same_except(_first(metamodel), model) &&
            return true
    end
    return false
end
