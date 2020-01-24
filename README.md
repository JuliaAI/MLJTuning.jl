# MLJTuning

Hyperparameter optimization for
[MLJ](https://github.com/alan-turing-institute/MLJ.jl) machine
learning models.

*Note:* This component of the [MLJ
  stack](https://github.com/alan-turing-institute/MLJ.jl#the-mlj-universe)
  applies to MLJ versions 0.8.0 and higher. Prior to 0.8.0, tuning
  algorithms resided in
  [MLJ](https://github.com/alan-turing-institute/MLJ.jl).


## Who is this repo for?

This repository is not intended for the general MLJ user but is:

- a dependency of the
  [MLJ](https://github.com/alan-turing-institute/MLJ.jl) machine
  learning platform, allowing MLJ users to perform a variety of
  hyperparameter optimization tasks
  
- a place for developers to integrate hyperparameter optimization
  algorithms (here called *tuning strategies*) into MLJ, either
  natively (by adding code to [/src/strategies](/src/strategies)) or
  by importing and implementing an interface provided by this repo
  
MLJTuning is a component of the MLJ
  [stack](https://github.com/alan-turing-institute/MLJ.jl#the-mlj-universe)
  which does not have
  [MLJModels](https://github.com/alan-turing-institute/MLJModels.jl)
  as a dependency (no ability to search and load registered MLJ
  models). It does however depend on
  [MLJBase](https://github.com/alan-turing-institute/MLJBase.jl) and,
  in particular, on the resampling functionality currently residing
  there.


## What's provided here?

This repository contains:

- a **tuning wrapper** called `TunedModel` for transforming arbitrary
  MLJ models into "self-tuning" ones - that is, into models which,
  when fit, automatically optimize a specified subset of the
  original hyperparameters, using training data resampling, before
  training the optimal model on all supplied data

- an abstract **tuning strategy interface** to allow developers to
  conveniently implement common hyperparameter optimization
  strategies, such as:

  - [x] search a list of explicitly specified models `list = [model1,
	model2, ...]`

  - [x] grid search
  
  - [ ] Latin hypercubes

  - [ ] random search

  - [ ] simulated annealing

  - [ ] Bayesian optimization using Gaussian processes

  - [ ] structured tree Parzen estimators

  - [ ] multi-objective (Pareto) optimization

  - [ ] genetic algorithms

  - [ ] AD-powered gradient descent methods 

- a selection of **implementations** of the tuning strategy interface,
  currently all those accessible from
  [MLJ](https://github.com/alan-turing-institute/MLJ.jl) itself.
  
- the code defining the MLJ functions `learning_curves!` and `learning_curve` as
  these are essentially one-dimensional grid searches


## Implementing a New Tuning Strategy

This document assumes familiarity with the [Evaluating Model
Performance](https://alan-turing-institute.github.io/MLJ.jl/dev/evaluating_model_performance/)
and [Performance
Measures](https://alan-turing-institute.github.io/MLJ.jl/dev/performance_measures/)
sections of the MLJ manual. Tuning itself, from the user's
perspective, is described in [Tuning
Models](https://alan-turing-institute.github.io/MLJ.jl/dev/tuning_models/).


### Overview

What follows is an overview of tuning in MLJ. After the overview is an
elaboration on those terms given in *italics*.

All tuning in MLJ is conceptualized as an iterative procedure, each
iteration corresponding to a performance *evaluation* of a single
*model*. Each such model is a mutation of a fixed *prototype*. In the
general case, this prototype is a composite model, i.e., a model with
other models as hyperparameters, and while the type of the prototype
mutations is fixed, the types of the sub-models are allowed to vary.

When all iterations of the algorithm are complete, the optimal model
is selected based entirely on a *history* generated according to the
specified *tuning strategy*. Iterations are generally performed in
batches, which are evaluated in parallel (sequential tuning strategies
degenerating into semi-sequential strategies, unless the batch size is
one). At the beginning of each batch, both the history and an internal
*state* object are consulted, and, on the basis of the tuning
strategy, a new batch of models to be evaluated is generated. On the
basis of these evaluations, and the strategy, the history and internal
state are updated.

The tuning algorithm initializes the state object before iterations
begin, on the basis of the specific strategy and a user-specified
*range* object.

- Recall that in MLJ a *model* is an object storing the
  hyperparameters of some learning algorithm indicated by the name of
  the model type (e.g., `DecisionTreeRegressor`). Models do not
  store learned parameters.

- An *evaluation* is the value returned by some call to the
  `evaluate!` method, when passed the resampling strategy (e.g.,
  `CV(nfolds=9)` and performance measures specified by the user when
  specifying the tuning task (e.g., `cross_entropy`,
  `accuracy`). Recall that such a value is a named tuple of vectors
  with keys `measure`, `measurement`, `per_fold`, and
  `per_observation`. See [Evaluating Model
  Performance](https://alan-turing-institute.github.io/MLJ.jl/dev/evaluating_model_performance/)
  for details. Recall also that some measures in MLJ (e.g.,
  `cross_entropy`) report a loss (or score) for each provided
  observation, while others (e.g., `auc`) report only an aggregated
  value (the `per_observation` entries being recorded as
  `missing`). This and other behavior can be inspected using trait
  functions. Do `info(rms)` to view the trait values for the `rms`
  loss, and see [Performance
  measures](https://alan-turing-institute.github.io/MLJ.jl/dev/performance_measures/)
  for details.

- The *history* is a vector of tuples generated by the tuning
  algorithm - one tuple per iteration - used to determine the optimal
  model and which also records other user-inspectable statistics that
  may be of interest - for example, evaluations of a measure (loss or
  score) different from one being explicitly optimized. Each tuple is
  of the form `(m, r)`, where `m` is a model instance and `r` is
  information
  about `m` extracted from an evaluation.

- A *tuning strategy* is an instance of some subtype `S <:
  TuningStrategy`, the name `S` (e.g., `Grid`) indicating the tuning
  algorithm to be applied. The fields of the tuning strategy - called
  *hyperparameters* - are those tuning parameters specific to the
  strategy that **do not refer to specific models or specific model
  hyperparameters**. So, for example, a default resolution to be used
  in a grid search is a hyperparameter of `Grid`, but the resolution
  to be applied to a *specific* hyperparameter (such as the maximum
  depth of a decision tree) is **not**. This latter parameter would be
  part of the user-specified range object.

- A *range* is any object whose specification completes the
  specification of the tuning task, after the prototype, tuning
  strategy, resampling strategy, performance measure(s), and total
  iteration count are given - and is essentially the space of models
  to be searched. This definition is intentionally broad and the
  interface places no restriction on the allowed types of this
  object. For the range objects supported by the `Grid` strategy, see
  [below](#range-types).


### Interface points for user input

Recall, for context, that in MLJ tuning is implemented as a model
wrapper. A model is tuned by *fitting* the wrapped model to data
(which also trains the optimal model on all available data). To use
the optimal model one *predicts* using the wrapped model. For more
detail, see the [Tuning
Models](https://alan-turing-institute.github.io/MLJ.jl/dev/tuning_models/)
section of the MLJ manual.

In setting up a tuning task, the user constructs an instance of the
`TunedModel` wrapper type, which has these principal fields:

- `model`: the prototype model instance mutated during tuning (the
  model being wrapped)

- `tuning`: the tuning strategy, an instance of a concrete
  `TuningStrategy` subtype, such as `Grid`

- `resampling`: the resampling strategy used for performance
  evaluations, which must be an instance of a concrete
  `ResamplingStrategy` subtype, such as `Holdout` or `CV`

- `measure`: a measure (loss or score) or vector of measures available
  to the tuning algorithm, the first of which is optimized in the
  common case of single-objective tuning strategies

- `range`: as defined above - roughly, the space of models to be searched

- `n`: the number of iterations (number of distinct models to be
  evaluated)

- `acceleration`: the computational resources to be applied (e.g.,
  `CPUProcesses()` for distributed computing and `CPUThreads()` for
  multi-threaded processing)

- `acceleration_resampling`: the computational resources to be applied
  at the level of resampling (e.g., in cross-validation)


### Implementation requirements for new tuning strategies

#### Summary of functions

Several functions are part of the tuning strategy API:

- `setup`: for initialization of state (compulsory)

- `result`: for building each element of the history

- `models!`: for generating batches of new models and updating the
  state (compulsory)

- `best`: for extracting the entry in the history corresponding to the
  optimal model from the full history

- `tuning_report`: for selecting what to report to the user apart from
  details on the optimal model

- `default_n`: to specify the number of models to be evaluated when
  `n` is not specified by the user

**Important note on the history.** The initialization and update of the
history is carried out internally, i.e., is not the responsibility of
the tuning strategy implementation. The history is always initialized to
`nothing`, rather than an empty vector.

The above functions are discussed further below, after discussing types.


#### The tuning strategy type

Each tuning algorithm must define a subtype of `TuningStrategy` whose
fields are the hyperparameters controlling the strategy that do not
directly refer to models or model hyperparameters. These would
include, for example, the default resolution of a grid search, or the
initial temperature in simulated annealing.

The algorithm implementation must include a keyword constructor with
defaults. Here's an example:

```julia
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
```

#### Range types

Generally new types are defined for each class of range object a
tuning strategy should like to handle, and the tuning strategy
functions to be implemented are dispatched on these types. Here are
the range objects supported by `Grid`:

  - one-dimensional `NumericRange` or `NominalRange` objects (these
  types are provided by MLJBase)

  - a tuple `(p, r)` where `p` is one of the above range objects, and
	`r` a resolution to override the default `resolution` of the
	strategy

  - vectors of objects of the above form, e.g., `[r1, (r2, 5), r3]`
	where `r1` and `r2` are `NumericRange` objects and `r3` a
	`NominalRange` object.

Recall that `NominalRange` has a `values` field, while `NominalRange`
has the fields `upper`, `lower`, `scale`, `unit` and `origin`. The
`unit` field specifies a preferred length scale, while `origin` a
preferred "central value". These default to `(upper - lower)/2` and
`(upper + lower)/2`, respectively, in the bounded case (neither `upper
= Inf` nor `lower = -Inf`). The fields `origin` and `unit` are used in
generating grids for unbounded ranges (and could be used in other
strategies for fitting two-parameter probability distributions, for
example).

A `ParamRange` object is always associated with the name of a
hyperparameter (a field of the prototype in the context of tuning)
which is recorded in its `field` attribute, but for composite models
this might be a be a "nested name", such as `:(atom.max_depth)`.


#### The `result` method: For declaring what parts of an evaluation goes into the history

```julia
MLJTuning.result(tuning::MyTuningStrategy, history, e)
```

This method is for extracting from an evaluation `e` of some model `m`
the value of `r` to be recorded in the corresponding tuple `(m, r)` of
the history. The value of `r` is also allowed to depend on previous
events in the history.

```julia
MLJTuning.result(tuning, history, e) = (measure=e.measure, measurement=e.measurement)
```

Note in this case that the result is always a named tuple of
*vectors*, since multiple measures can be specified (and singleton
measures provided by the user are promoted to vectors with a
single element).

The history must contain everything needed for the `best` method to
determine the optimal model, and everything needed by the
`report_history` method, which generates a report on tuning to the
user (for use in visualization, for example). These methods are
detailed below.


#### The `setup` method: To initialize state

```julia
state = setup(tuning::MyTuningStrategy, model, range, verbosity)
```

The `setup` function is for initializing the `state` of the tuning
algorithm (needed, by the algorithm's `models!` method; see below). Be
sure to make this object mutable if it needs to be updated by the
`models!` method. The `state` generally stores, at the least, the
range or some processed version thereof. In momentum-based gradient
descent, for example, the state would include the previous
hyperparameter gradients, while in GP Bayesian optimization, it would
store the (evolving) Gaussian processes.

If a variable is to be reported as part of the user-inspectable
history, then it should be written to the history instead of stored in
state. An example of this might be the `temperature` in simulated
annealing.

The `verbosity` is an integer indicating the level of logging: `0`
means logging should be restricted to warnings, `-1`, means completely
silent.

The fallback for `setup` is:

```julia
setup(tuning::TuningStrategy, model, range, verbosity) = range
```

However, a tuning strategy will generally want to implement a `setup`
method for each range type it is going to support:

```julia
MLJTuning.setup(tuning::MyTuningStrategy, model, range::RangeType1, verbosity) = ...
MLJTuning.setup(tuning::MyTuningStrategy, model, range::RangeType2, verbosity) = ...
etc.
```


#### The `models!` method: For generating model batches to evaluate

```julia
MLJTuning.models!(tuning::MyTuningStrategy, model, history, state, verbosity)
```

This is the core method of a new implementation. Given the existing
`history` and `state`, it must return a vector ("batch") of *new*
model instances to be evaluated. Any number of models can be returned
(and this includes an empty vector or `nothing`, if models have been
exhausted) and the evaluations will be performed in parallel (using
the mode of parallelization defined by the `acceleration` field of the
`TunedModel` instance). *An update of the history, performed
automatically under the hood, only occurs after these evaluations.*

Most sequential tuning strategies will want include the batch size as
a hyperparameter, which we suggest they call `batch_size`, but this
field is not part of the tuning interface. In tuning, whatever models
are returned by `models!` get evaluated in parallel.

In a `Grid` tuning strategy, for example, `models!` returns a random
selection of `n - length(history)` models from the grid, so that
`models!` is called only once (in each call to
`MLJBase.fit(::TunedModel, ...)` or `MLJBase.update(::TunedModel,
...)`). In a bona fide sequential method which is generating models
non-deterministically (such as simulated annealing), `models!` might
return a single model, or return a small batch of models to make use
of parallelization (the method becoming "semi-sequential" in that
case). In sequential methods that generate new models
deterministically (such as those choosing models that optimize the
expected improvement of a surrogate statistical model) `models!` would
return a single model.

If the tuning algorithm exhausts it's supply of new models (because,
for example, there is only a finite supply) then `models!` should
return an empty vector. Under the hood, there is no fixed "batch-size"
parameter, and the tuning algorithm is happy to receive any number
of models.


#### The `best` method: To define what constitutes the "optimal model"

```julia
MLJTuning.best(tuning::MyTuningStrategy, history)
```

Returns the entry `(best_model, r)` from the history corresponding to
the optimal model `best_model`.

A fallback whose definition is given below may be used, *provided the
fallback for `result` detailed above has not been overloaded*. In this
fallback for `best`, the best model is the one optimizing performance
estimates for the first measure in the `TunedModel` field `measure`:

```julia
function best(tuning::TuningStrategy, history)
   measurements = [h[2].measurement[1] for h in history]
   measure = first(history)[2].measure[1]
   if orientation(measure) == :score
	   measurements = -measurements
   end
   best_index = argmin(measurements)
   return history[best_index]
end
```

####  The `tuning_report` method: To build the user-accessible report

As with any model, fitting a `TunedModel` instance generates a
user-accessible report. In the case of tuning, the report is
constructed with this code:

```julia
report = merge((best_model=best_model, best_result=best_result, best_report=best_report,),
				tuning_report(tuning, history, state))
```

where:

- `best_model` is the optimal model instance

- `best_result` is the corresponding "result" entry in the history (e.g., performance evaluation)

- `best_report` is the report generated by fitting the optimal
model

- `tuning_report(::MyTuningStrategy, ...)` is a method the implementer
  may overload. It should return a named tuple. The fallback is to
  return the raw history:

```julia
MLJTuning.tuning_report(tuning, history, state) = (history=history,)
```

#### The `default_n` method: For declaring the default number of iterations

```julia
MLJTuning.default_n(tuning::MyTuningStrategy, range)
```

The `methods!` method (which is allowed to return multiple models) is
called until a history of length `n` has been built, or `models!`
returns an empty list or `nothing`. If the user does not specify a
value for `n` when constructing her `TunedModel` object, then `n` is
set to `default_n(tuning, range)` at construction, where `range` is
the user specified range.

The fallback is

```julia
MLJTuning.default_n(::TuningStrategy, range) = 10
```


### Implementation example: Search through an explicit list

The most rudimentary tuning strategy just evaluates every model in a
specified list of models sharing a common type, such lists
constituting the only kind of supported range. (In this special case
`range` is an arbitrary iterator of models, which are `Probabilistic`
or `Deterministic`, according to the type of the prototype `model`,
which is otherwise ignored.) The fallback implementations for `setup`,
`result`, `best` and `report_history` suffice.  In particular, there
is not distinction between `range` and `state` in this case.

Here's the complete implementation:

```julia

import MLJBase

mutable struct Explicit <: TuningStrategy end

# models! returns all available models in the range at once:
MLJTuning.models!(tuning::Explicit, model, history::Nothing,
				  state, verbosity) = state
MLJTuning.models!(tuning::Explicit, model, history,
				  state, verbosity) = state[length(history) + 1:end]

function MLJTuning.default_n(tuning::Explicit, range)
	try
		length(range)
	catch MethodError
		10
	end
end
```

For slightly less trivial example, see
[/src/strategies/grid.jl](/src/strategies/grid.jl)
