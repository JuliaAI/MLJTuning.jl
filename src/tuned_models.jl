## TYPES AND CONSTRUCTOR

mutable struct DeterministicTunedModel{T,M<:Deterministic,R} <: MLJBase.Deterministic
    model::M
    tuning::T  # tuning strategy
    resampling # resampling strategy
    measure
    weights::Union{Nothing,Vector{<:Real}}
    operation
    range::R
    train_best::Bool
    repeats::Int
    n::Union{Int,Nothing}
    acceleration::AbstractResource
    acceleration_resampling::AbstractResource
    check_measure::Bool
end

mutable struct ProbabilisticTunedModel{T,M<:Probabilistic,R} <: MLJBase.Probabilistic
    model::M
    tuning::T  # tuning strategy
    resampling # resampling strategy
    measure
    weights::Union{Nothing,AbstractVector{<:Real}}
    operation
    range::R
    train_best::Bool
    repeats::Int
    n::Union{Int,Nothing}
    acceleration::AbstractResource
    acceleration_resampling::AbstractResource
    check_measure::Bool
end

const EitherTunedModel{T,M} =
    Union{DeterministicTunedModel{T,M},ProbabilisticTunedModel{T,M}}

MLJBase.is_wrapper(::Type{<:EitherTunedModel}) = true

#todo update:
"""
    tuned_model = TunedModel(; model=nothing,
                             tuning=Grid(),
                             resampling=Holdout(),
                             measure=nothing,
                             weights=nothing,
                             repeats=1,
                             operation=predict,
                             range=nothing,
                             n=default_n(tuning, range),
                             train_best=true,
                             acceleration=default_resource(),
                             acceleration_resampling=CPU1(),
                             check_measure=true)

Construct a model wrapper for hyperparameter optimization of a
supervised learner.

Calling `fit!(mach)` on a machine `mach=machine(tuned_model, X, y)` or
`mach=machine(tuned_model, X, y, w)` will:

- Instigate a search, over clones of `model`, with the hyperparameter
  mutations specified by `range`, for a model optimizing the specified
  `measure`, using performance evaluations carried out using the
  specified `tuning` strategy and `resampling` strategy.

- Fit an internal machine, based on the optimal model
  `fitted_params(mach).best_model`, wrapping the optimal `model`
  object in *all* the provided data `X, y` (or in `task`). Calling
  `predict(mach, Xnew)` then returns predictions on `Xnew` of this
  internal machine. The final train can be supressed by setting
  `train_best=false`.

The `range` objects supported depend on the `tuning` strategy
specified. Query the `strategy` docstring for details. To optimize
over an explicit list `v` of models of the same type, use
`strategy=Explicit()` and specify `model=v[1]` and `range=v`.

If `measure` supports weights (`supports_weights(measure) == true`)
then any `weights` specified will be passed to the measure. If more
than one `measure` is specified, then only the first is optimized
(unless `strategy` is multi-objective) but the performance against
every measure specified will be computed and reported in
`report(mach).best_performance` and other relevant attributes of the
generated report.

Specify `repeats > 1` for repeated resampling per model evaluation. See
[`evaluate!](@ref) options for details.

*Important.* If a custom measure `measure` is used, and the measure is
a score, rather than a loss, be sure to check that
`MLJ.orientation(measure) == :score` to ensure maximization of the
measure, rather than minimization. Override an incorrect value with
`MLJ.orientation(::typeof(measure)) = :score`.

*Important:* If `weights` are left unspecified, and `measure` supports
sample weights, then any weight vector `w` used in constructing a
corresponding tuning machine, as in `tuning_machine =
machine(tuned_model, X, y, w)` (which is then used in *training* each
model in the search) will also be passed to `measure` for evaluation.

In the case of two-parameter tuning, a Plots.jl plot of performance
estimates is returned by `plot(mach)` or `heatmap(mach)`.

Once a tuning machine `mach` has bee trained as above, one can access
the learned parameters of the best model, using
`fitted_params(mach).best_fitted_params`. Similarly, the report of
training the best model is accessed via `report(mach).best_report`.

"""
function TunedModel(;model=nothing,
                    tuning=Grid(),
                    resampling=MLJBase.Holdout(),
                    measures=nothing,
                    measure=measures,
                    weights=nothing,
                    operation=predict,
                    ranges=nothing,
                    range=ranges,
                    train_best=true,
                    repeats=1,
                    n=default_n(tuning, range),
                    acceleration=default_resource(),
                    acceleration_resampling=CPU1(),
                    check_measure=true)

    range === nothing && error("You need to specify `range=...` unless "*
                               "`tuning isa Explicit`. ")
    model == nothing && error("You need to specify model=... .\n"*
                              "If `tuning=Explicit()`, any model in the "*
                              "range will do. ")

    if model isa Deterministic
        tuned_model = DeterministicTunedModel(model, tuning, resampling,
                                       measure, weights, operation, range,
                                              train_best, repeats, n,
                                              acceleration,
                                              acceleration_resampling,
                                              check_measure)
    elseif model isa Probabilistic
        tuned_model = ProbabilisticTunedModel(model, tuning, resampling,
                                       measure, weights, operation, range,
                                              train_best, repeats, n,
                                              acceleration,
                                              acceleration_resampling,
                                              check_measure)
    else
        error("Only `Deterministic` and `Probabilistic` "*
              "model types supported.")
    end

    message = clean!(tuned_model)
    isempty(message) || @info message

    return tuned_model

end

function MLJBase.clean!(tuned_model::EitherTunedModel)
    message = ""
    if tuned_model.measure === nothing
        tuned_model.measure = default_measure(tuned_model.model)
        if tuned_model.measure === nothing
            error("Unable to deduce a default measure for specified model. "*
                  "You must specify `measure=...`. ")
        else
            message *= "No measure specified. "*
            "Setting measure=$(tuned_model.measure). "
        end
    end
    if !(tuned_model.acceleration isa Union{CPU1, CPUProcesses})
        message *= "Supported `acceleration` types are `CPU1` "*
        "and `CPUProcesses`. Setting `acceleration=CPU1()`. "
        tuned_model.acceleration = CPU1()
    end
    return message
end


## FIT AND UPDATE METHODS

# returns a (model, result) pair for the history:
function event(model, resampling_machine, verbosity, tuning, history)
    resampling_machine.model.model = model
    verb = (verbosity == 2 ? 0 : verbosity - 1)
    fit!(resampling_machine, verbosity=verb)
    e = evaluate(resampling_machine)
    r = result(tuning, history, e)

    if verbosity > 2
        println(params(model))
    end
    if verbosity > 1
        println("$r")
    end

    return deepcopy(model), r
end

function assemble_events(models, resampling_machine,
                         verbosity, tuning, history, acceleration::CPU1)
    map(models) do m
        event(m, resampling_machine, verbosity, tuning, history)
    end
end

function assemble_events(models, resampling_machine,
                         verbosity, tuning, history, acceleration::CPUProcesses)
    pmap(models) do m
        event(m, resampling_machine, verbosity, tuning, history)
    end
end

# history is intialized to `nothing` because it's type is not known.
_vcat(history, Δhistory) = vcat(history, Δhistory)
_vcat(history::Nothing, Δhistory) = Δhistory
_length(history) = length(history)
_length(::Nothing) = 0

# builds on an existing `history` until the length is `n` or the model
# supply is exhausted(method shared by `fit` and `update`). Returns
# the bigger history:
function build(history, n, tuning, model::M,
               state, verbosity, acceleration, resampling_machine) where M
    j = _length(history)
    models_exhausted = false
    while j < n && !models_exhausted
        _models = models!(tuning, model, history, state, verbosity)
        models = _models === nothing ? M[] : collect(_models)
        Δj = length(models)
        Δj == 0 && (models_exhausted = true)
        shortfall = n - Δj
        if models_exhausted && shortfall > 0 && verbosity > -1
            @warn "Only $j < n = $n`  models evaluated.\n"*
            "Model supply prematurely exhausted. "
        end
        Δj == 0 && break
        shortfall < 0 && (models = models[1:n - j])
        j += Δj

        Δhistory = assemble_events(models, resampling_machine,
                                 verbosity, tuning, history, acceleration)
        history = _vcat(history, Δhistory)
    end
    return history
end

function MLJBase.fit(tuned_model::EitherTunedModel{T,M},
                     verbosity::Integer, data...) where {T,M}
    tuning = tuned_model.tuning
    n = tuned_model.n
    model = tuned_model.model
    range = tuned_model.range
    n === Nothing && (n = default_n(tuning, range))
    acceleration = tuned_model.acceleration

    state = setup(tuning, model, range, verbosity)

    # instantiate resampler (`model` to be replaced with mutated
    # clones during iteration below):
    resampler = Resampler(model=model,
                          resampling    = tuned_model.resampling,
                          measure       = tuned_model.measure,
                          weights       = tuned_model.weights,
                          operation     = tuned_model.operation,
                          check_measure = tuned_model.check_measure,
                          repeats       = tuned_model.repeats,
                          acceleration  = tuned_model.acceleration_resampling)
    resampling_machine = machine(resampler, data...)

    history = build(nothing, n, tuning, model, state,
                    verbosity, acceleration, resampling_machine)

    best_model, best_result = best(tuning, history)
    fitresult = machine(best_model, data...)

    if tuned_model.train_best
        fit!(fitresult, verbosity=verbosity - 1)
        prereport = (best_model=best_model, best_result=best_result,
                     best_report=MLJBase.report(fitresult))
    else
        prereport = (best_model=best_model, best_result=best_result,
                     best_report=missing)
    end

    report = merge(prereport, tuning_report(tuning, history, state))
    meta_state = (history, deepcopy(tuned_model), state, resampling_machine)

    return fitresult, meta_state, report
end

function MLJBase.update(tuned_model::EitherTunedModel, verbosity::Integer,
                        old_fitresult, old_meta_state, data...)

    history, old_tuned_model, state, resampling_machine = old_meta_state

    n = tuned_model.n
    acceleration = tuned_model.acceleration

    if MLJBase.is_same_except(tuned_model, old_tuned_model, :n)

        tuning=tuned_model.tuning
        model=tuned_model.model

        if tuned_model.n > old_tuned_model.n
            # temporarily mutate tuned_model:
            tuned_model.n = n - old_tuned_model.n

            history = build(history, n, tuning, model, state,
                            verbosity, acceleration, resampling_machine)

            # restore tuned_model to original state
            tuned_model.n = n
        else
            verbosity < 1 || @info "Number of tuning iterations `n` "*
            "lowered.\nTruncating existing tuning history and "*
            "retraining new best model."
        end
        best_model, best_result = best(tuning, history)

        fitresult = machine(best_model, data...)

        if tuned_model.train_best
            fit!(fitresult, verbosity=verbosity - 1)
            prereport = (best_model=best_model, best_result=best_result,
                         best_report=MLJBase.report(fitresult))
        else
            prereport = (best_model=best_model, best_result=best_result,
                         best_report=missing)
        end

        _report = merge(prereport, tuning_report(tuning, history, state))

        meta_state = (history, deepcopy(tuned_model), state)

        return fitresult, meta_state, _report

    else

        return fit(tuned_model, verbosity, data...)

    end

end

MLJBase.predict(tuned_model::EitherTunedModel, fitresult, Xnew) =
    predict(fitresult, Xnew)

function MLJBase.fitted_params(tuned_model::EitherTunedModel, fitresult)
    if tuned_model.train_best
        return (best_model=fitresult.model,
                best_fitted_params=fitted_params(fitresult))
    else
        return (best_model=fitresult.model,
                best_fitted_params=missing)
    end
end


## METADATA

MLJBase.supports_weights(::Type{<:EitherTunedModel{<:Any,M}}) where M =
    MLJBase.supports_weights(M)

MLJBase.load_path(::Type{<:DeterministicTunedModel}) =
    "MLJTuning.DeterministicTunedModel"
MLJBase.package_name(::Type{<:EitherTunedModel}) = "MLJTuning"
MLJBase.package_uuid(::Type{<:EitherTunedModel}) = "MLJTuning"
MLJBase.package_url(::Type{<:EitherTunedModel}) =
    "https://github.com/alan-turing-institute/MLJTuning.jl"
MLJBase.is_pure_julia(::Type{<:EitherTunedModel{T,M}}) where {T,M} =
    MLJBase.is_pure_julia(M)
MLJBase.input_scitype(::Type{<:EitherTunedModel{T,M}}) where {T,M} =
    MLJBase.input_scitype(M)
MLJBase.target_scitype(::Type{<:EitherTunedModel{T,M}}) where {T,M} =
    MLJBase.target_scitype(M)

