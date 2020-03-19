mutable struct Explicit <: TuningStrategy end 

# models! returns all available models in the range at once:
function MLJTuning.models!(tuning::Explicit,
                           model,
                           history,
                           state,
                           verbosity)
    history === nothing && return state
    return state[length(history) + 1:end]
end

