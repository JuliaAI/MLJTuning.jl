"""
LatinHypercube(nGenerations = 1, popSize = 100, nTournament = 2,
                pTournament = 0.8. interSampleWeight = 1.0,
                ae_power = 2, periodic_ae = false)

Instantiate  grid-based hyperparameter tuning strategy using the library
LatinHypercubeSampling.jl. The optimised Latin Hypercube sampling plan is
created using a genetic based optimization algorithm based on the inverse of the
Audze-Eglais function.
The optimization is run for nGenerations. The population size, number of samples
selected, probability of selection, the inter sample weight of sampling and the
norm can be choosen. There is also the possibility of using a periodic version
of the Audze-Eglais which reduces clustering along the boundaries of the
sampling plan. To enable this feature set `periodic_ae = true`.

### Supported ranges:

A single one-dimensional range or vector of one-dimensioinal ranges
can be specified. Specifically, in `LatinHypercubeSampling` search,
the `range` field of a `TunedModel` instance can be:

- A single one-dimensional range - ie, `ParamRange` object - `r`, constructed
using the `range` method.

- Any vector of objects of the above form

"""
mutable struct LatinHypercube <: TuningStrategy
    nGenerations::Int
    popSize::Int
    nTournament::Int
    pTournament::Number
    interSampleWeight::Number
    ae_power::Number
    periodic_ae::Bool
    rng::Random.AbstractRNG
end

function LatinHypercube(; nGenerations = 1, popSize = 100, nTournament = 2,
                        pTournament = 0.8, interSampleWeight = 1.0,
                        ae_power = 2, periodic_ae = false,
                        rng=Random.GLOBAL_RNG)

               _rng = rng isa Integer ? Random.MersenneTwister(rng) : rng

              return LatinHypercube(nGenerations,popSize,nTournament,
                                    pTournament, interSampleWeight, ae_power,
                                    periodic_ae, _rng)

end

function _create_bounds_and_dims(d,r)
    bounds = []
    dims = []
    for i = 1:d
        if r[i] isa NumericRange
            if !(r[i].scale isa Symbol)
                error("Callable scale not supported in LatinHyperCube tuning.")
            end
            push!(dims,LatinHypercubeSampling.Continuous())
            if isfinite(r[i].lower) && isfinite(r[i].upper)
                push!(bounds,(transform(MLJBase.Scale,MLJBase.scale(r[i].scale),r[i].lower),
                 transform(MLJBase.Scale,MLJBase.scale(r[i].scale),r[i].upper)))
            elseif !isfinite(r[i].lower) && isfinite(r[i].upper)
                push!(bounds,(transform(MLJBase.Scale,MLJBase.scale(r[i].scale),r[i].upper - 2*r[i].unit),
                 transform(MLJBase.Scale,MLJBase.scale(r[i].scale),r[i].upper)))
            elseif isfinite(r[i].lower) && !isfinite(r[i].upper)
                push!(bounds,(transform(MLJBase.Scale,MLJBase.scale(r[i].scale),r[i].lower),
                 transform(MLJBase.Scale,MLJBase.scale(r[i].scale),r[i].lower + 2*r[i].unit)))
            else
                push!(bounds,(transform(MLJBase.Scale,MLJBase.scale(r[i].scale),r[i].origin - r[i].unit),
                 transform(MLJBase.Scale,MLJBase.scale(r[i].scale),r[i].origin + r[i].unit)))
            end
        else
            push!(dims, LatinHypercubeSampling.Categorical(length(r[i].values), 1.0))
            push!(bounds,(0,length(r[i].values)))
        end
    end
    return bounds, dims
end

function setup(tuning::LatinHypercube, model, r, verbosity)
    d = length(r)
    bounds, dims = _create_bounds_and_dims(d, r)
    initial_plan = randomLHC(n, dims, nGenerations,
                             popsize = tuning.popSize,
                             ntour = tuning.nTournament,
                             ptour = tuning.pTournament,
                             interSampleWeight = tuning.interSampleWeight,
                             periodic_ae = tuning.periodic_ae,
                             ae_power = tuning.ae_power,
                             rng = tuning.rng)
    scaled_plan = scaleLHC(initial_plan, bounds)
    @inbounds for i = 1:size(scaled_plan,1)
        for j = 1:size(scaled_plan,2)
            if dims[j] isa LatinHypercubeSampling.Continuous
                scaled_plan[i][j] = inverse_transform(MLJBase.Scale,
                                                      MLJBase.scale(r[j].scale),
                                                      scaled_plan[i][j])
            else
                scaled_plan[i][j] = r[j].values[scaled_plan[i][j]]
            end
        end
    end
    ranges = r
    fields = map(r -> r.field, ranges)
    models = makeLatinHypercube(model, fields, scaled_plan)
    state = (models = models,
             fields = fields)
    return state
end

function MLJTuning.models(tuning::LatinHypercube,
                          model,
                          hystory,
                          state,
                          n_remaining,
                          verbosity)
     return state.models[_length(history) + 1:end], state
end

function makeLatinHypercube(prototype::Model,fields,plan)
    N = size(plan,1)
    map(1:N) do i
        clone = deepcopy(prototype)
        for k in eachindex(fields)
            recursive_setproperty(clone,fields[k],plan[i,k])
        end
        clone
    end
end
