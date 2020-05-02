## LEARNING CURVES

"""
    curve = learning_curve(mach; resolution=30,
                                 resampling=Holdout(),
                                 repeats=1,
                                 measure=rms,
                                 weights=nothing,
                                 operation=predict,
                                 range=nothing,
                                 acceleration=default_resource(),
                                 acceleration_grid=CPU1(),
                                 rngs=nothing,
                                 rng_name=nothing)

Given a supervised machine `mach`, returns a named tuple of objects
suitable for generating a plot of performance estimates, as a function
of the single hyperparameter specified in `range`. The tuple `curve`
has the following keys: `:parameter_name`, `:parameter_scale`,
`:parameter_values`, `:measurements`.

To generate multiple curves for a `model` with a random number
generator (RNG) as a hyperparameter, specify the name, `rng_name`, of
the (possibly nested) RNG field, and a vector `rngs` of RNG's, one for
each curve. Alternatively, set `rngs` to the number of curves desired,
in which case RNG's are automatically generated. The individual curve
computations can be distributed across multiple processes using
`acceleration=CPUProcesses()`. See the second example below for a
demonstration.

```julia
X, y = @load_boston;
atom = @load RidgeRegressor pkg=MultivariateStats
ensemble = EnsembleModel(atom=atom, n=1000)
mach = machine(ensemble, X, y)
r_lambda = range(ensemble, :(atom.lambda), lower=10, upper=500, scale=:log10)
curve = learning_curve(mach; range=r_lambda, resampling=CV(), measure=mav)
using Plots
plot(curve.parameter_values,
     curve.measurements,
     xlab=curve.parameter_name,
     xscale=curve.parameter_scale,
     ylab = "CV estimate of RMS error")
```

If using a `Holdout()` `resampling` strategy (with no shuffling) and
if the specified hyperparameter is the number of iterations in some
iterative model (and that model has an appropriately overloaded
`MLJModelInterface.update` method) then training is not restarted from scratch
for each increment of the parameter, ie the model is trained
progressively.

```julia
atom.lambda=200
r_n = range(ensemble, :n, lower=1, upper=250)
curves = learning_curve(mach; range=r_n, verbosity=0, rng_name=:rng, rngs=3)
plot!(curves.parameter_values,
     curves.measurements,
     xlab=curves.parameter_name,
     ylab="Holdout estimate of RMS error")


```
    learning_curve(model::Supervised, X, y; kwargs...)
    learning_curve(model::Supervised, X, y, w; kwargs...)

Plot a learning curve (or curves) directly, without first constructing
a machine.

"""
learning_curve(mach::Machine{<:Supervised}; kwargs...) =
    learning_curve(mach.model, mach.args...; kwargs...)

# for backwards compatibility
learning_curve!(mach::Machine{<:Supervised}; kwargs...) =
    learning_curve(mach; kwargs...)

function learning_curve(model::Supervised, args...;
                        resolution=30,
                        resampling=Holdout(),
                        weights=nothing,
                        measures=nothing,
                        measure=measures,
                        operation=predict,
                        ranges::Union{Nothing,ParamRange}=nothing,
                        range::Union{Nothing,ParamRange},
                        repeats=1,
                        acceleration=default_resource(),
                        acceleration_grid=CPU1(),
                        verbosity=1,
                        rngs=nothing,
                        rng_name=nothing,
                        check_measure=true)

    range !== nothing || error("No param range specified. Use range=... ")

    if rngs != nothing
        rng_name == nothing &&
            error("Having specified `rngs=...`, you must specify "*
                  "`rng_name=...` also. ")
        if rngs isa Integer
            rngs = MersenneTwister.(1:rngs)
        elseif rngs isa AbstractRNG
            rngs = [rngs, ]
        elseif !(rngs isa AbstractVector{<:AbstractRNG})
            error("`rng` must have type `Integer` , `AbstractRNG` or "*
                  "`AbstractVector{<:AbstractRNG}`. ")
        end
    end

    tuned_model = TunedModel(model=model,
                             range=range,
                             tuning=Grid(resolution=resolution,
                                         shuffle=false),
                             resampling=resampling,
                             operation=operation,
                             measure=measure,
                             train_best=false,
                             weights=weights,
                             repeats=repeats,
                             acceleration=acceleration_grid)

    tuned = machine(tuned_model, args...)
    ## One tuned_mach per thread
    tuned_machs = Dict(1 => tuned)

    results = _tuning_results(rngs, acceleration, tuned_machs, rng_name, verbosity)

    parameter_name=results.parameter_names[1]
    parameter_scale=results.parameter_scales[1]
    parameter_values=[results.parameter_values[:, 1]...]
    measurements = results.measurements

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
_tuning_results(rngs, acceleration, tuned_machs, rngs_name, verbosity) =
    error("acceleration=$acceleration unsupported. ")

# single curve:
_tuning_results(rngs::Nothing, acceleration, tuned_machs, rngs_name, verbosity) =
    _single_curve(tuned_machs[1], verbosity)

function _single_curve(tuned, verbosity)
    fit!(tuned, verbosity=verbosity, force=true)
    tuned.report.plotting
end

# CPU1:
function _tuning_results(rngs::AbstractVector, acceleration::CPU1,
                         tuned_machs, rng_name, verbosity)
    tuned = tuned_machs[1]
    old_rng = recursive_getproperty(tuned.model.model, rng_name)
    ret = reduce(_collate,
                 [begin
                  recursive_setproperty!(tuned.model.model, rng_name, rng)
                  fit!(tuned, verbosity=verbosity, force=true)
                  tuned.report.plotting
                  end
                  for rng in rngs])

    recursive_setproperty!(tuned.model.model, rng_name, old_rng)

    return ret
end

# CPUProcesses:
function _tuning_results(rngs::AbstractVector, acceleration::CPUProcesses,
    tuned_machs, rng_name, verbosity)
    tuned = tuned_machs[1]
    old_rng = recursive_getproperty(tuned.model.model, rng_name)
    ret = @distributed (_collate) for rng in rngs
        recursive_setproperty!(tuned.model.model, rng_name, rng)
        fit!(tuned, verbosity=-1, force=true)
        tuned.report.plotting
    end
    recursive_setproperty!(tuned.model.model, rng_name, old_rng)

    return ret
end

# CPUThreads:
@static if VERSION >= v"1.3.0-DEV.573"
function _tuning_results(rngs::AbstractVector, acceleration::CPUThreads,
    tuned_machs, rng_name, verbosity)
    
    n_threads = Threads.nthreads()
    if n_threads == 1
        return _tuning_results(rngs, CPU1(),
                         tuned_machs, rng_name, verbosity)
    end
    
    n_rngs = length(rngs)
    old_rng = recursive_getproperty(tuned_machs[1].model.model, rng_name)

    results = Array{Any, 1}(undef, n_rngs)
   
    @sync begin  
      
      @sync for rng_part in Iterators.partition(1:n_rngs, max(1,floor(Int, n_rngs/n_threads)))    
        Threads.@spawn begin
          foreach(rng_part) do k
            id = Threads.threadid()
            if !haskey(tuned_machs, id)
                ## deepcopy of model is because other threads can still change the state
                ## of tuned_machs[id].model.model
                   tuned_machs[id] =
                       machine(TunedModel(model = deepcopy(tuned_machs[1].model.model),
                             range=tuned_machs[1].model.range,
                             tuning=tuned_machs[1].model.tuning,
                             resampling=tuned_machs[1].model.resampling,
                             operation=tuned_machs[1].model.operation,
                             measure=tuned_machs[1].model.measure,
                             train_best=tuned_machs[1].model.train_best,
                             weights=tuned_machs[1].model.weights,
                             repeats=tuned_machs[1].model.repeats,
                             acceleration=tuned_machs[1].model.acceleration),
                             tuned_machs[1].args...)
            end
            recursive_setproperty!(tuned_machs[id].model.model, rng_name, rngs[k]) 
            fit!(tuned_machs[id], verbosity=-1, force=true)
            results[k] = tuned_machs[id].report.plotting   
          end
        end
    end
    recursive_setproperty!(tuned_machs[1].model.model, rng_name, old_rng)
   end
  
   return reduce(_collate, results)

end

end
