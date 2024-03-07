const WARN_INCONSISTENT_PREDICTION_TYPE =
    "Not all models to be evaluated have the same prediction type, and this may "*
    "cause problems for some measures. For example, a probabilistic metric "*
    "like `log_loss` cannot be applied to a model making point (deterministic) "*
    "predictions. Inspect the prediction type with "*
        "`prediction_type(model)`. "

mutable struct Explicit <: TuningStrategy end

struct ExplicitState{R, N}
    range::R # a model-generating iterator
    next::N  # to hold output of `iterate(range)`
    prediction_type::Symbol
    user_warned::Bool
end

function MLJTuning.setup(tuning::Explicit, model, range, n, verbosity)
    next = iterate(range)
    return ExplicitState(range, next, MLJBase.prediction_type(model), false)
end

# models! returns as many models as possible but no more than `n_remaining`:
function MLJTuning.models(tuning::Explicit,
                          model,
                          history,
                          state,
                          n_remaining,
                          verbosity)

    range, next, prediction_type, user_warned =
        state.range, state.next, state.prediction_type, state.user_warned

    function check(m)
        if !user_warned && verbosity > -1 && MLJBase.prediction_type(m) != prediction_type
            @warn WARN_INCONSISTENT_PREDICTION_TYPE
            user_warned = true
        end
    end

    next === nothing && return nothing, state

    m, s = next
    check(m)

    models = Any[m, ] # types not known until run-time

    next = iterate(range, s)

    i = 1 # current length of `models`
    while i < n_remaining
        next === nothing && break
        m, s = next
        check(m)
        push!(models, m)
        i += 1
        next = iterate(range, s)
    end

    new_state = ExplicitState(range, next, prediction_type, user_warned)

    return models, new_state

end

function default_n(tuning::Explicit, range)
    try
        length(range)
    catch MethodError
        DEFAULT_N
    end
end
