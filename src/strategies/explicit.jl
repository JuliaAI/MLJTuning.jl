const ERR_INCONSISTENT_PREDICTION_TYPE = ArgumentError(
    "Not all models to be evaluated have the same prediction type. Inspect "*
        "these with `prediction_type(model)`. "
)

mutable struct Explicit <: TuningStrategy end

struct ExplicitState{R, N}
    range::R # a model-generating iterator
    next::N # to hold output of `iterate(range)`
    prediction_type::Symbol
end

ExplictState(r::R, n::N) where {R,N} = ExplicitState{R, Union{Nothing, N}}(r, n)

function MLJTuning.setup(tuning::Explicit, model, range, n, verbosity)
    next = iterate(range)
    return ExplicitState(range, next, MLJBase.prediction_type(model))
end

# models! returns as many models as possible but no more than `n_remaining`:
function MLJTuning.models(tuning::Explicit,
                          model,
                          history,
                          state,
                          n_remaining,
                          verbosity)

    range, next, prediction_type  = state.range, state.next, state.prediction_type

    check = ==(prediction_type)

    next === nothing && return nothing, state

    m, s = next
    check(MLJBase.prediction_type(m)) || throw(ERR_INCONSISTENT_PREDICTION_TYPE)

    models = Any[m, ] # types not known until run-time

    next = iterate(range, s)

    i = 1 # current length of `models`
    while i < n_remaining
        next === nothing && break
        m, s = next
        check(MLJBase.prediction_type(m)) ||
            throw(ERR_INCONSISTENT_PREDICTION_TYPE)
        push!(models, m)
        i += 1
        next = iterate(range, s)
    end

    new_state = ExplicitState(range, next, prediction_type)

    return models, new_state

end

function default_n(tuning::Explicit, range)
    try
        length(range)
    catch MethodError
        DEFAULT_N
    end
end
