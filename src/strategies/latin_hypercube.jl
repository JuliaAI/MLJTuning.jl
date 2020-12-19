"""
LatinHypercube(n_max = MLJTuning.DEFAULT_N,
               nGenerations = 1,
               popSize = 100,
               nTournament = 2,
               pTournament = 0.8.,
               interSampleWeight = 1.0,
               ae_power = 2,
               periodic_ae = false,
               rng=Random.GLOBAL_RNG)

Instantiate grid-based hyperparameter tuning strategy using the
library [LatinHypercubeSampling.jl](https://github.com/MrUrq/LatinHypercubeSampling.jl).

An optimised Latin Hypercube sampling plan is created using a genetic
based optimization algorithm based on the inverse of the Audze-Eglais
function.  The optimization is run for `nGenerations` and creates a
maximum number of `n_max` points for evaluation (A `TunedModel`
instance can specify any `n < n_max`).

To use a periodic version of the Audze-Eglais function (to reduce
clustering along the boundaries) specify `periodic_ae = true`.

### Supported ranges:

A single one-dimensional range or vector of one-dimensioinal ranges
can be specified. Specifically, in `LatinHypercubeSampling` search,
the `range` field of a `TunedModel` instance can be:

- A single one-dimensional range - ie, `ParamRange` object - `r`, constructed
using the `range` method.

- Any vector of objects of the above form

"""
mutable struct LatinHypercube <: TuningStrategy
    n_max::Int
    gens::Int
    popsize::Int
    ntour::Int
    ptour::Number
    interSampleWeight::Number
    ae_power::Number
    periodic_ae::Bool
    rng::Random.AbstractRNG
end


function LatinHypercube(; n_max = DEFAULT_N, gens = 1,
                        popsize = 100, ntour = 2, ptour = 0.8,
                        interSampleWeight = 1.0, ae_power = 2,
                        periodic_ae = false,rng=Random.GLOBAL_RNG)

    _rng = rng isa Integer ? Random.MersenneTwister(rng) : rng

    return LatinHypercube(n_max, gens, popsize, ntour,
                          ptour, interSampleWeight, ae_power,
                          periodic_ae, _rng)

end

function _create_bounds_and_dims_type(d,r)
    bounds = []
    dims_type = Array{LatinHypercubeSampling.LHCDimension}(undef,0)
    for i = 1:d
        if r[i] isa NumericRange
            if !(r[i].scale isa Symbol)
                error("Callable scale not supported.")
            end
            push!(dims_type,LatinHypercubeSampling.Continuous())
            if isfinite(r[i].lower) && isfinite(r[i].upper)
                push!(bounds,
                      Float64.([transform(MLJBase.Scale,
                                          MLJBase.scale(r[i].scale),
                                          r[i].lower),
                                transform(MLJBase.Scale,
                                          MLJBase.scale(r[i].scale),
                                          r[i].upper)]))
            elseif !isfinite(r[i].lower) && isfinite(r[i].upper)
                push!(bounds,
                      Float64.([transform(MLJBase.Scale,
                                          MLJBase.scale(r[i].scale),
                                          r[i].upper - 2*r[i].unit),
                                transform(MLJBase.Scale,
                                          MLJBase.scale(r[i].scale),
                                          r[i].upper)]))
            elseif isfinite(r[i].lower) && !isfinite(r[i].upper)
                push!(bounds,Float64.([transform(MLJBase.Scale,
                                                 MLJBase.scale(r[i].scale),
                                                 r[i].lower),
                                       transform(MLJBase.Scale,
                                                 MLJBase.scale(r[i].scale),
                                                 r[i].lower + 2*r[i].unit)]))
            else
                push!(bounds,Float64.([transform(MLJBase.Scale,
                                                 MLJBase.scale(r[i].scale),
                                                 r[i].origin - r[i].unit),
                                       transform(MLJBase.Scale,
                                                 MLJBase.scale(r[i].scale),
                                                 r[i].origin + r[i].unit)]))
            end
        else
            push!(dims_type,
                  LatinHypercubeSampling.Categorical(length(r[i].values),
                                                     1.0))
            push!(bounds,Float64.([1,length(r[i].values)]))
        end
    end
    return Tuple.(bounds), dims_type
end

function setup(tuning::LatinHypercube, model, r, verbosity)
    d = length(r)
    bounds, dims_type = _create_bounds_and_dims_type(d, r)
    plan, _ = LatinHypercubeSampling.LHCoptim(tuning.n_max, d, tuning.gens,
                    rng = tuning.rng,
                    popsize = tuning.popsize,
                    ntour = tuning.ntour,
                    ptour = tuning.ptour,
                    dims = dims_type,
                    interSampleWeight = tuning.interSampleWeight,
                    periodic_ae = tuning.periodic_ae,
                    ae_power = tuning.ae_power)
    scaled_plan = LatinHypercubeSampling.scaleLHC(plan, bounds)
    for i = 1:size(scaled_plan,1)
        for j = 1:size(scaled_plan,2)
            if dims_type[j] isa LatinHypercubeSampling.Continuous
                if r[j] isa MLJBase.NumericRange{Int,MLJBase.Bounded,Symbol}
                    scaled_plan[i,j] =
                        Int(floor(inverse_transform(MLJBase.Scale,
                                                    MLJBase.scale(r[j].scale),
                                                    scaled_plan[i,j])))
                end
            else
                scaled_plan[i,j] = r[j].values[Int(scaled_plan[i,j])]
            end
        end
    end
    ranges = r
    fields = map(r -> r.field, ranges)
    parameter_scales = scale.(r)
    models = makeLatinHypercube(model, fields, scaled_plan)
    state = (models=models,
             fields=fields,
             parameter_scales=parameter_scales)
    return state
end

function MLJTuning.models(tuning::LatinHypercube,
                          model,
                          history,
                          state,
                          n_remaining,
                          verbosity)
     return state.models[_length(history) + 1:end], state
end

tuning_report(tuning::LatinHypercube, history, state) =
    (plotting = plotting_report(state.fields, state.parameter_scales, history),)

function makeLatinHypercube(prototype::Model,fields,plan)
    N = size(plan,1)
    map(1:N) do i
        clone = deepcopy(prototype)
        for k in eachindex(fields)
            recursive_setproperty!(clone,fields[k],plan[i,k])
        end
        clone
    end
end
