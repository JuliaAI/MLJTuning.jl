const ParameterName=Union{Symbol,Expr}

"""
    RandomSearch(bounded=Distributions.Uniform,
                 positive_unbounded=Distributions.Gamma,
                 other=Distributions.Normal,
                 rng=Random.GLOBAL_RNG)

Instantiate a random search tuning strategy, for searching over
Cartesian hyperparameter domains, with customizable priors in each
dimension.

### Supported ranges

A single one-dimensional range or vector of one-dimensioinal ranges
can be specified. If not paired with a prior, then one is fitted,
according to fallback distribution types specified by the tuning
strategy hyperparameters. Specifically, in `RandomSearch`, the `range`
field of a `TunedModel` instance can be:

- a single one-dimensional range (`ParamRange` object) `r`

- a pair of the form `(r, d)`, with `r` as above and where `d` is:

    - a probability vector of the same length as `r.values` (`r` a
      `NominalRange`)

    - any `Distributions.UnivariateDistribution` *instance* (`r` a
      `NumericRange`)

    - one of the *subtypes* of `Distributions.UnivariateDistribution`
      listed in the table below, for automatic fitting using
      `Distributions.fit(d, r)`, a distribution whose support always
      lies between `r.lower` and `r.upper` (`r` a `NumericRange`)

- any pair of the form `(field, s)`, where `field` is the (possibly
  nested) name of a field of the model to be tuned, and `s` an
  arbitrary sampler object for that field. This means only that
  `rand(rng, s)` is defined and returns valid values for the field.

- any vector of objects of the above form

A range vector may contain multiple entries for the same model field,
as in `range = [(:lambda, s1), (:alpha, s), (:lambda, s2)]`. In that
case the entry used in each iteration is random.

distribution types  | for fitting to ranges of this type
--------------------|-----------------------------------
`Arcsine`, `Uniform`, `Biweight`, `Cosine`, `Epanechnikov`, `SymTriangularDist`, `Triweight` | bounded
`Gamma`, `InverseGaussian`, `Poisson` | positive (bounded or unbounded)
`Normal`, `Logistic`, `LogNormal`, `Cauchy`, `Gumbel`, `Laplace`  | any

`ParamRange` objects are constructed using the `range` method.

### Examples

    using Distributions

    range1 = range(model, :hyper1, lower=0, upper=1)

    range2 = [(range(model, :hyper1, lower=1, upper=10), Arcsine),
              range(model, :hyper2, lower=2, upper=Inf, unit=1, origin=3),
              (range(model, :hyper2, lower=2, upper=4), Normal(0, 3)),
              (range(model, :hyper3, values=[:ball, :tree]), [0.3, 0.7])]

    # uniform sampling of :(atom.λ) from [0, 1] without defining a NumericRange:
    struct MySampler end
    Base.rand(rng::Random.AbstractRNG, ::MySampler) = rand(rng)
    range3 = (:(atom.λ), MySampler())

### Algorithm

In each iteration, a model is generated for evaluation by mutating the
fields of a deep copy of `model`. The range vector is shuffled and the
fields sampled according to the new order (repeated fields being
mutated more than once). For a `range` entry of the form `(field, s)`
the algorithm calls `rand(rng, s)` and mutates the field `field` of
the model clone to have this value. For an entry of the form `(r, d)`,
`s` is substituted with `sampler(r, d)`. If no `d` is specified, then
sampling is uniform (with replacement) if `r` is a `NominalRange`, and
is otherwise given by the defaults specified by the tuning strategy
parameters `bounded`, `positive_unbounded`, and `other`, depending on
the field values of the `NumericRange` object `r`.

See also [`TunedModel`](@ref), [`range`](@ref), [`sampler`](@ref).

"""
mutable struct RandomSearch <: TuningStrategy
    bounded
    positive_unbounded
    other
    rng::Random.AbstractRNG
end

# Constructor with keywords
function RandomSearch(; bounded=Distributions.Uniform,
                      positive_unbounded=Distributions.Gamma,
                      other=Distributions.Normal,
                      rng=Random.GLOBAL_RNG)
    (bounded isa Type{<:Distributions.UnivariateDistribution} &&
        positive_unbounded isa Type{<:Distributions.UnivariateDistribution} &&
        other isa Type{<:Distributions.UnivariateDistribution}) ||
        error("`bounded`, `positive_unbounded` and `other` "*
              "must all be subtypes of "*
              "`Distributions.UnivariateDistribution`. ")

    _rng = rng isa Integer ? Random.MersenneTwister(rng) : rng
    return RandomSearch(bounded, positive_unbounded, other, _rng)
end

# `state` consists of a tuple of (field, sampler) pairs (that gets
# shuffled each iteration):
setup(tuning::RandomSearch, model, user_range, verbosity) =
    process_random_range(user_range,
                              tuning.bounded,
                              tuning.positive_unbounded,
                              tuning.other) |> collect

function MLJTuning.models(tuning::RandomSearch,
                          model,
                          history,
                          state, # tuple of (field, sampler) pairs
                          n_remaining,
                          verbosity)
    new_models = map(1:n_remaining) do _
        clone = deepcopy(model)
        Random.shuffle!(tuning.rng, state)
        for (fld, s) in state
            recursive_setproperty!(clone, fld, rand(tuning.rng, s))
        end
        clone
    end
    return new_models, state
end

function tuning_report(tuning::RandomSearch, history, field_sampler_pairs)

    fields = first.(field_sampler_pairs)
    parameter_scales = map(field_sampler_pairs) do (fld, s)
        scale(s)
    end

    plotting = plotting_report(fields, parameter_scales, history)

    return (plotting=plotting,)

end
