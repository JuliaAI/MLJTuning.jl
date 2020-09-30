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
sampling plan. This feature can be specified by the periodic_ae variable.

### Supported ranges:

A single one-dimensional range or vector of one-dimensioinal ranges
can be specified. Specifically, in `LatinHypercubeSampling` search,
the `range` field of a `TunedModel` instance can be:

- A single one-dimensional range - ie, `ParamRange` object - `r`,

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

function setup(tuning::LatinHypercube, model, user_range, verbosity)
    d = length(user_range)
    dim_matrix = zeros(d,2)
    #need to take into account other types of ranges (Nominal)
    for i = 1:d
        if isfinite(r[i].lower) && isfinite(r[i].upper)
            dim_matrix[i,1] = r[i].lower
            dim_matrix[i,2] = r[i].upper
        elseif !isfinite(r[i].lower) && isfinite(r[i].upper)
            dim_matrix[i,1] = r[i].upper - 2*r.unit
            dim_matrix[i,2] = r[i].upper
        elseif isfinite(r[i].lower) && !isfinite(r[i].upper)
            dim_matrix[i,1] = r[i].lower
            dim_matrix[i,2] = r[i].lower + 2*r.unit
        else
            dim_matrix[i,1] = r.origin - r.unit
            dim_matrix[i,2] = r.origin + r.unit
        end
    end
    bounds = [Tuple(x[i,:]) for i = 1:d]

    plan, _ = LHCoptim(n,d,nGenerations,
                      popsize = popSize,
                      ntour = nTournament,
                      ptour = pTournment,
                      interSampleWeight = interSampleWeight,
                      periodic_ae = periodic_ae,
                      ae_power = ae_power)

    scaled_plan = scaleLHC(plan,bounds)

    ranges = user_range
    fields = map(r -> r.field, ranges)
    fields = unique(fields)

    models = makeLatinHypercube(model,fields,plan)
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
