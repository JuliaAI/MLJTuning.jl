## TYPES TO BE SUBTYPED

abstract type TuningStrategy <: MLJBase.MLJType end
MLJBase.show_as_constructed(::Type{<:TuningStrategy}) = true


## METHODS TO BE IMPLEMENTED

# for validating and resetting invalid fields in tuning strategy
MLJBase.clean!(tuning::TuningStrategy) = ""

# for initialization of state (compulsory)
setup(tuning::TuningStrategy, model, range, n, verbosity) = range

# for adding extra user-inspectable information to the history:
extras(tuning::TuningStrategy, history, state, E) = NamedTuple()

# for generating batches of new models and updating the state (but not
# history):
function models end

# for adding to the default report:
tuning_report(tuning::TuningStrategy, history, state) = NamedTuple()

# for declaring the default number of models to evaluate:
default_n(tuning::TuningStrategy, range) = DEFAULT_N

# for encoding the selection_heuristics supported by given strategy:
supports_heuristic(heuristic::Any, strategy::Any) = false
