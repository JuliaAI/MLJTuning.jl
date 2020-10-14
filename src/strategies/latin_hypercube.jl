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
end

LatinHypercube(; nGenerations = 1, popSize = 100, nTournament = 2,
               pTournament = 0.8, interSampleWeight = 1.0,
               ae_power = 2, periodic_ae = false) =
              LatinHypercube(nGenerations,popSize,nTournament,pTournament)

function setup(tuning::LatinHypercube, model, r, verbosity)
    d = length(r)
    dim_matrix = zeros(d,2)
    dims = []
    bounds = map(dim_matrix) do r
        if r isa NumericRange
            push!(dims,Continuous())
            if isfinite(r.lower) && isfinite(r.upper)
                (transform(MLJBase.Scale,scale(r.scale),r.lower),
                 transform(MLJBase.Scale,scale(r.scale),r.upper))
            elseif !isfinite(r.lower) && isfinite(r.upper)
                (transform(MLJBase.Scale,scale(r.scale),r.upper - 2*r.unit),
                 transform(MLJBase.Scale,scale(r.scale),r.upper))
            elseif isfinite(r.lower) && !isfinite(r.upper)
                (transform(MLJBase.Scale,scale(r.scale),r.lower),
                 transform(MLJBase.Scale,scale(r.scale),r.lower + 2*r.unit))
            else
                (transform(MLJBase.Scale,scale(r.scale),r.origin - r.unit),
                 transform(MLJBase.Scale,scale(r.scale),r.origin + r.unit))
            end
        else
            push!(dims, Categorical(length(r.values)))
            (0,length(r.values))
        end

    end
    initial_plan = randomLHC(n,dims,nGenerations,
                              popsize = popSize,
                              ntour = nTournament,
                              ptour = pTournment,
                              interSampleWeight = interSampleWeight,
                              periodic_ae = periodic_ae,
                              ae_power = ae_power)
    scaled_plan = scaleLHC(initial_plan, bounds)

    #inverse_transforming the values in case the dims is continuous,
    #mapping to right symbol in case it is nominal
    @inbounds for i = 1:size(scaled_plan,1)
        for j = 1:size(scaled_plan,2)
            if dims[j] isa LatinHypercubeSampling.Continuous
                scaled_plan[i][j] = inverse_transform(MLJBase.Scale,
                                                      scale(r[j].scale),
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
