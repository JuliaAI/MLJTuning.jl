const ParameterName=Union{Symbol,Expr}

"""
    Grid(goal=nothing, resolution=10, rng=Random.GLOBAL_RNG, shuffle=true)

Instantiate a Cartesian grid-based hyperparameter tuning strategy with
a specified number of grid points as `goal`, or using a specified
default `resolution` in each numeric dimension.

### Supported ranges:

- A single one-dimensional range (`ParamRange` object) `r`, or pair of
  the form `(r, res)` where `res` specifies a resolution to override
  the default `resolution`.

- Any vector of objects of the above form

`ParamRange` objects are constructed using the `range` method.

Example 1:

    range(model, :hyper1, lower=1, origin=2, unit=1)

Example 2:

    [(range(model, :hyper1, lower=1, upper=10), 15),
      range(model, :hyper2, lower=2, upper=4),
      range(model, :hyper3, values=[:ball, :tree]]

Note: All the `field` values of the `ParamRange` objects (`:hyper1`,
`:hyper2`, `:hyper3` in the preceding example) must refer to field
names a of single model (the `model` specified during `TunedModel`
construction).


### Algorithm

This is a standard grid search with the following specifics: In all
cases all `values` of each specified `NominalRange` are exhausted. If
`goal` is specified, then all resolutions are ignored, and a global
resolution is applied to the `NumericRange` objects that maximizes the
number of grid points, subject to the restriction that this not exceed
`goal`. Otherwise the default `resolution` and any parameter-specific
resolutions apply.

In all cases the models generated are shuffled using `rng`, unless
`shuffle=false`.

See also [TunedModel](@ref), [range](@ref).

"""
mutable struct Grid <: TuningStrategy
    goal::Union{Nothing,Int}
    resolution::Int
    shuffle::Bool
    rng::Random.AbstractRNG
end

# Constructor with keywords
Grid(; goal=nothing, resolution=10, shuffle=true,
     rng=Random.GLOBAL_RNG) =
    Grid(goal, resolution, MLJBase.shuffle_and_rng(shuffle, rng)...)

isnumeric(::Any) = false
isnumeric(::NumericRange) = true

adjusted_resolutions(::Nothing,  ranges, resolutions) = resolutions
function adjusted_resolutions(goal, ranges, resolutions)
    # get the product Π of the lengths of the NominalRanges:
    len(::NumericRange) = 1
    len(r::NominalRange) = length(r.values)
    Π = prod(len.(ranges))

    n_numeric = sum(isnumeric.(ranges))

    # compute new uniform resolution:
    goal = goal/Π
    res = round(Int, goal^(1/n_numeric))
    return  map(eachindex(resolutions)) do j
        isnumeric(ranges[j]) ? res : resolutions[j]
    end
end

function setup(tuning::Grid, model, user_range, verbosity)
    ranges, resolutions =
        process_user_range(user_range, tuning.resolution, verbosity)
    resolutions = adjusted_resolutions(tuning.goal, ranges, resolutions)

    fields = map(r -> r.field, ranges)

    parameter_scales = scale.(ranges)

    if tuning.shuffle
        models = grid(tuning.rng, model, ranges, resolutions)
    else
        models = grid(model, ranges, resolutions)
    end

    state = (models=models,
             fields=fields,
             parameter_scales=parameter_scales)

    return state

end

MLJTuning.models!(tuning::Grid, model, history::Nothing,
                  state, verbosity) = state.models
MLJTuning.models!(tuning::Grid, model, history,
                  state, verbosity) =
    state.models[length(history) + 1:end]

function tuning_report(tuning::Grid, history, state)

    plotting = plotting_report(state.fields, state.parameter_scales, history)

    # todo: remove collects?
    return (history=history, plotting=plotting)

end

function default_n(tuning::Grid, user_range)
    ranges, resolutions =
        process_user_range(user_range, tuning.resolution, -1)

    resolutions = adjusted_resolutions(tuning.goal, ranges, resolutions)
    len(t::Tuple{NumericRange,Integer}) = length(iterator(t[1], t[2]))
    len(t::Tuple{NominalRange,Integer}) = t[2]
    return prod(len.(zip(ranges, resolutions)))

end
