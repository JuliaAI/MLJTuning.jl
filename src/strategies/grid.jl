const ParameterName=Union{Symbol,Expr}

"""
    Grid(goal=nothing, resolution=10, rng=Random.GLOBAL_RNG, shuffle=true)

Instantiate a Cartesian grid-based hyperparameter tuning strategy with
a specified number of grid points as `goal`, or using a specified
default `resolution` in each numeric dimension.

### Supported ranges:

A single one-dimensional range or vector of one-dimensioinal ranges
can be specified. Specifically, in `Grid` search, the `range` field
of a `TunedModel` instance can be:

- A single one-dimensional range - ie, `ParamRange` object - `r`, or
  pair of the form `(r, res)` where `res` specifies a resolution to
  override the default `resolution`.

- Any vector of objects of the above form

Two elements of a `range` vector may share the same `field` attribute,
with the effect that their grids are combined, as in Example 3 below.

`ParamRange` objects are constructed using the `range` method.


Example 1:

    range(model, :hyper1, lower=1, origin=2, unit=1)

Example 2:

    [(range(model, :hyper1, lower=1, upper=10), 15),
      range(model, :hyper2, lower=2, upper=4),
      range(model, :hyper3, values=[:ball, :tree])]

Example 3:

    # a range generating the grid `[1, 2, 10, 20, 30]` for `:hyper1`:
    [range(model, :hyper1, values=[1, 2]),
     (range(model, :hyper1, lower= 10, upper=30), 3)]

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
`goal`. (This assumes no field appears twice in the `range` vector.)
Otherwise the default `resolution` and any parameter-specific
resolutions apply.

In all cases the models generated are shuffled using `rng`, unless
`shuffle=false`.

See also [`TunedModel`](@ref), [`range`](@ref).

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

# To replace resolutions for numeric ranges with goal-adjusted ones if
# a goal is specified:
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

# For deciding scale for duplicated fields:
_merge(s1, s2) = (s1 == :none ? s2 : s1)

function fields_iterators_and_scales(ranges, resolutions)

    # following could have non-unique entries:
    fields = map(r -> r.field, ranges)

    iterator_given_field = Dict{Union{Symbol,Expr},Vector}()
    scale_given_field = Dict{Union{Symbol,Expr},Any}()
    for i in eachindex(ranges)
        fld = fields[i]
        r = ranges[i]
        if haskey(iterator_given_field, fld)
            iterator_given_field[fld] =
                vcat(iterator_given_field[fld], iterator(r, resolutions[i]))
            scale_given_field[fld] =
                _merge(scale_given_field[fld], scale(r))
        else
            iterator_given_field[fld] = iterator(r, resolutions[i])
            scale_given_field[fld] = scale(r)
        end
    end
    fields = unique(fields)
    iterators = map(fld->iterator_given_field[fld], fields)
    scales = map(fld->scale_given_field[fld], fields)

    return fields, iterators, scales

end

function setup(tuning::Grid, model, user_range, n, verbosity)
    ranges, resolutions =
        process_grid_range(user_range, tuning.resolution, verbosity)

    resolutions = adjusted_resolutions(tuning.goal, ranges, resolutions)

    fields, iterators, parameter_scales =
        fields_iterators_and_scales(ranges, resolutions)

    if tuning.shuffle
        models = grid(tuning.rng, model, fields, iterators)
    else
        models = grid(model, fields, iterators)
    end

    state = (models=models,
             fields=fields,
             parameter_scales=parameter_scales,
             models_delivered=false)

    return state

end

# models returns all models on first call:
function MLJTuning.models(tuning::Grid,
                          model,
                          history,
                          state,
                          n_remaining,
                          verbosity)
    state.models_delivered && return nothing, state
    state = (models=state.models,
             fields=state.fields,
             parameter_scales=state.parameter_scales,
             models_delivered=true)
    return state.models, state
end

tuning_report(tuning::Grid, history, state) =
    (plotting = plotting_report(state.fields, state.parameter_scales, history),)

function default_n(tuning::Grid, user_range)

    ranges, resolutions =
        process_grid_range(user_range, tuning.resolution, -1)

    resolutions = adjusted_resolutions(tuning.goal, ranges, resolutions)

    fields, iterators, parameter_scales =
        fields_iterators_and_scales(ranges, resolutions)

    return prod(length.(iterators))

end
