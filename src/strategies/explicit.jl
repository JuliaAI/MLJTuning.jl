mutable struct Explicit <: TuningStrategy end

struct ExplicitState{R,N}
    range::R # a model-generating iterator
    next::N # to hold output of `iterate(range)`
end

ExplicitState(r::R, ::Nothing) where R = ExplicitState{R,Nothing}(r,nothing)
ExplictState(r::R, n::N) where {R,N} = ExplicitState{R,Union{Nothing,N}}(r,n)

function MLJTuning.setup(tuning::Explicit, model, range, verbosity)
    next = iterate(range)
    return ExplicitState(range, next)
end

# models! returns all available models in the range at once:
function MLJTuning.models(tuning::Explicit,
                           model,
                           history,
                           state,
                           n_remaining,
                           verbosity)

    range, next  = state.range, state.next

    next === nothing && return nothing, state

    m, s = next
    models = [m, ]

    next = iterate(range, s)

    i = 1 # current length of `models`
    while i < n_remaining
        next === nothing && break
        m, s = next
        push!(models, m)
        i += 1
        next = iterate(range, s)
    end

    new_state = ExplicitState(range, next)

    return models, new_state

end

function default_n(tuning::Explicit, range)
    try
        length(range)
    catch MethodError
        DEFAULT_N
    end
end
