abstract type TuningStrategy <: MLJBase.MLJType end
MLJBase.show_as_constructed(::Type{<:TuningStrategy}) = true

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
function default_n(tuning::TuningStrategy, range)
    try
        length(range)
    catch MethodError
        DEFAULT_N
    end
end

