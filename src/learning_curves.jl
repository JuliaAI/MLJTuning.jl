## LEARNING CURVES

"""
    curve = learning_curve!(mach; resolution=30,
                                  resampling=Holdout(),
                                  repeats=1,
                                  measure=rms,
                                  weights=nothing,
                                  operation=predict,
                                  range=nothing,
                                  acceleration=default_resource(),
                                  acceleration_grid=CPU1(),
                                  n_curves=1)

Given a supervised machine `mach`, returns a named tuple of objects
suitable for generating a plot of performance estimates, as a function
of the single hyperparameter specified in `range`. The tuple `curve`
has the following keys: `:parameter_name`, `:parameter_scale`,
`:parameter_values`, `:measurements`.

For `n_curves > 1`, multiple curves are computed, and the value of
`curve.measurements` is an array, one column for each run. This is
useful in the case of models with indeterminate fit-results, such as a
random forest. The curve computations can be distributed across
multiple processors using `acceleration=CPUProcesses()` but it is the
responsibility of the user to seed the relevant random number
generators separately for each process. Otherwise identical curves
will result.

````julia
X, y = @load_boston;
atom = @load RidgeRegressor pkg=MultivariateStats
ensemble = EnsembleModel(atom=atom, n=1000)
mach = machine(ensemble, X, y)
r_lambda = range(ensemble, :(atom.lambda), lower=10, upper=500, scale=:log10)
curve = learning_curve!(mach; range=r_lambda, resampling=CV(), measure=mav)
using Plots
plot(curve.parameter_values,
     curve.measurements,
     xlab=curve.parameter_name,
     xscale=curve.parameter_scale,
     ylab = "CV estimate of RMS error")
````

If using a `Holdout` `resampling` strategy, and the specified
hyperparameter is the number of iterations in some iterative model
(and that model has an appropriately overloaded `MLJBase.update`
method) then training is not restarted from scratch for each increment
of the parameter, ie the model is trained progressively.

````julia
atom.lambda=200
r_n = range(ensemble, :n, lower=1, upper=250)
curves = learning_curve!(mach; range=r_n, verbosity=0, n_curves=5)
plot(curves.parameter_values,
     curves.measurements,
     xlab=curves.parameter_name,
     ylab="Holdout estimate of RMS error")
````

"""
function learning_curve!(mach::Machine{<:Supervised};
                         resolution=30,
                         resampling=Holdout(),
                         weights=nothing,
                         measure=nothing,
                         operation=predict,
                         range::Union{Nothing,ParamRange}=nothing,
                         repeats=1,
                         acceleration=default_resource(),
                         acceleration_grid=CPU1(),
                         verbosity=1,
                         n=1,
                         n_curves=n,
                         check_measure=true)

    if measure == nothing
        measure = default_measure(mach.model)
        verbosity < 1 ||
            @info "No measure specified. Using measure=$measure. "
    end

    range !== nothing || error("No param range specified. Use range=... ")

    tuned_model = TunedModel(model=mach.model, ranges=range,
                             tuning=Grid(resolution=resolution,
                                         shuffle=false),
                             resampling=resampling,
                             operation=operation,
                             measure=measure,
                             train_best=false,
                             weights=weights,
                             repeats=repeats,
                             acceleration=acceleration_grid)

    tuned = machine(tuned_model, mach.args...)

    results = _tuning_results(acceleration, tuned, n_curves, verbosity)

    parameter_name=results.parameter_names[1]
    parameter_scale=results.parameter_scales[1]
    parameter_values=[results.parameter_values[:, 1]...]
    measurements = results.measurements
    measurements = (n_curves == 1) ? [measurements...] : measurements

    return (parameter_name=parameter_name,
            parameter_scale=parameter_scale,
            parameter_values=parameter_values,
            measurements=measurements)
end

_collate(plotting1, plotting2) =
    merge(plotting1,
          (measurements=hcat(plotting1.measurements,
                             plotting2.measurements),))

# fallback:
_tuning_results(acceleration, tuned, verbosity) =
    error("acceleration=$acceleration unsupported. ")

# CPU1:
function _tuning_results(::CPU1, tuned, n_curves, verbosity)
    reduce(_collate,
           [(fit!(tuned, verbosity=verbosity, force=true);
             tuned.report.plotting) for j in 1:n_curves])
end

# CPUProcesses:
function _tuning_results(::CPUProcesses, tuned, n_curves, verbosity)
    @distributed (_collate) for j in 1:n_curves
        fit!(tuned, verbosity=-1, force=true)
        tuned.report.plotting
    end
end




"""
    learning_curve(model::Supervised, args...; kwargs...)

Plot a learning curve (or curves) without first constructing a
machine. Equivalent to `learing_curve!(machine(model, args...);
kwargs...)

See [learning_curve!](@ref)

"""
learning_curve(model::Supervised, args...; kwargs...) =
    learning_curve!(machine(model, args...); kwargs...)
