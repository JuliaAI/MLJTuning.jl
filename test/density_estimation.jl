using Test
using MLJTuning
using MLJBase
using StatisticalMeasures
using StableRNGs
import MLJModelInterface
import StatisticalMeasures: CategoricalDistributions, Distributions


# We define a density estimator to fit a `UnivariateFinite` distribution to some
# Categorical data, with a Laplace smoothing option, α.

mutable struct UnivariateFiniteFitter <: MLJModelInterface.Probabilistic
    alpha::Float64
end
UnivariateFiniteFitter(;alpha=1.0) = UnivariateFiniteFitter(alpha)

function MLJModelInterface.fit(model::UnivariateFiniteFitter,
                               verbosity, X, y)

    α = model.alpha
    N = length(y)
    _classes = classes(y)
    d = length(_classes)

    frequency_given_class = Distributions.countmap(y)
    prob_given_class =
        Dict(c => (get(frequency_given_class, c, 0) + α)/(N + α*d) for c in _classes)

    fitresult = CategoricalDistributions.UnivariateFinite(prob_given_class)

    report = (params=Distributions.params(fitresult),)
    cache = nothing

    verbosity > 0 && @info "Fitted a $fitresult"

    return fitresult, cache, report
end

MLJModelInterface.predict(model::UnivariateFiniteFitter,
                          fitresult,
                          X) = fitresult


MLJModelInterface.input_scitype(::Type{<:UnivariateFiniteFitter}) =
    Nothing
MLJModelInterface.target_scitype(::Type{<:UnivariateFiniteFitter}) =
    AbstractVector{<:Finite}

# This test will fail if MLJ test dependency MLJBase is < 1.11
@testset "tuning for density estimators" begin
    y = coerce(collect("abbabbc"), Multiclass)
    X = nothing

    train, test = partition(eachindex(y), 3/7)
    # For above train-test split, hand calculation determines, when optimizing against
    # log loss, that:
    best_alpha = 2.0
    best_loss = (4log(9) - log(3) - 2log(4) - log(2))/4

    model = UnivariateFiniteFitter(alpha=0)
    r = range(model, :alpha, values=[0.1, 1, 1.5, 2, 2.5, 10])
    tmodel = TunedModel(
        model,
        tuning=Grid(shuffle=false),
        range=r,
        resampling=[(train, test),],
        measure=log_loss,
        compact_history=false,
    )

    mach = machine(tmodel, X, y)
    fit!(mach, verbosity=0)
    best = report(mach).best_history_entry
    @test best.model.alpha == best_alpha
    @test best.evaluation.measurement[1] ≈ best_loss
end

true
