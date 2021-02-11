# Includes stopping criterion surveyed in Prechelt, Lutz (1998):
# "Early Stopping - But When?", in "Neural Networks: Tricks of the
# Trade", ed. G. Orr, Springer.

# https://link.springer.com/chapter/10.1007%2F3-540-49430-8_3

# name in paper | object
# --------------|----------------------
# GL_α          | wip
# PQ_α          | wip
# UP_s          | `Patience(n=...)`

# Also implemented:

# `Never()`
# `TimeLimit(t=...)`

const PRECHELT_REF = "[Prechelt, Lutz (1998): \"Early Stopping"*
    "- But When?\", in *Neural Networks: Tricks of the Trade*, "*
    "ed. G. Orr, Springer.](https://link.springer.com/chapter"*
    "/10.1007%2F3-540-49430-8_3)"

const STOPPING_DOC = "A stopping crieterion for use in tuning "*
    "and in training iterative models."

## HELPERS

# extract a loss from a history entry (reversing sign if first
# measurement is a score):
_loss(entry) = MLJTuning.signature(first(entry.measure))*
    first(entry.measurement)


## ABSTRACT TYPE

abstract type StoppingCriterion <: MLJBase.MLJType end
MLJBase.show_as_constructed(::Type{<:StoppingCriterion}) = true


## FALL BACK METHODS

for_history(::StoppingCriterion, fitted_params, reports, history) = nothing
stopping_early(::StoppingCriterion, history) = false


## NEVER

"""
    Never()

$STOPPING_DOC

Indicates early stopping is to be disabled.

"""
struct Never <: StoppingCriterion end


## TIME LIMIT

"""
    TimeLimit(; t=0.5)

$STOPPING_DOC

Stopping is triggered after `t` hours have elapsed, as measured between
timestamps written to the model evaluation history immediately after
each model performance evaluation (generally lower than the true
elapsed wall clock time).

Any Julia built-in `Real` type can be used for `t`, which is always
rounded to nearest millisecond internally. Subtypes of `Period` may
also be used, as in `TimeLimit(t=Minute(30))`.

"""
struct TimeLimit <: StoppingCriterion
    t::Millisecond
    function TimeLimit(t::Millisecond)
        t > Millisecond(0) ||
            throw(ArgumentError("Time limit `t` must be positive. "))
        return new(t)
    end
end
TimeLimit(t::T) where T <: Period = TimeLimit(convert(Millisecond, t))
# for t::T a "numeric" time in hours; assumes `round(Int, ::T)` implemented:
TimeLimit(t) = TimeLimit(round(Int, 3_600_000*t) |> Millisecond)
TimeLimit(; t =Minute(30)) = TimeLimit(t)

for_history(criterion::TimeLimit, ::Any, ::Any, ::Any) = now()
function stopping_early(criterion::TimeLimit, history)
    history === nothing && return false
    history[end].stopping_data - history[1].stopping_data > criterion.t
end


## GENERALIZATION LOSS

# This is GL_α in Prechelt 1998

"""
    GeneralizationLoss(; alpha=2.0)

$STOPPING_DOC

A stop is triggered when the *generalization loss* exceeds
the threshold `alpha`. The generalization loss for a sequence `E_1,
E_2, ..., E_t` of out-of-sample estimates of the loss is

`` GL = 100*(E_t - E_opt)/|E_opt|``

where `E_opt` is the minimum value of the sequence. In the case that
scores are estimated, their signs are reversed.

Denoted "GL_α" in $PRECHELT_REF.

"""
struct GeneralizationLoss <: StoppingCriterion
    alpha::Float64
    function GeneralizationLoss(alpha)
        alpha > 0 ||
            throw(ArgumentError("Threshold `alpha` must be positive. "))
        return new(alpha)
    end
end
GeneralizationLoss(; alpha=2.0) = GeneralizationLoss(alpha)

_E_opt(history) = minimum(map(history) do entry
                          _loss(entry)
                          end)
_GL(history) = 100*(_loss(last(history))/abs(_E_opt(history)) - 1)

stopping_early(criterion::GeneralizationLoss, history) =
    _GL(history) > criterion.alpha


## PATIENCE

# This is UP_s in Prechelt 1998

"""
    Patience(; n=5)

$STOPPING_DOC

A stop is triggered by `n` consecutive deteriorations in the out-of-sample
performance.

Denoted "_UP_s" in $PRECHELT_REF.

"""
mutable struct Patience <: StoppingCriterion
    n::Int
    function Patience(n::Int)
        n > 0 ||
            throw(ArgumentError("The patience level `n` must be positive. "))
        return new(n)
    end
end
Patience(; n=1) = Patience(n)

# recursive definition from Prechelt 1998:
function _UP(n, history)
    t = length(history)
    previous_loss = t ==1 ? Inf : _loss(history[t - 1])
    stop = _loss(history[t]) > previous_loss
    n == 1 && return stop
    return stop && _UP(n - 1, view(history, 1:(t - 1)))
end

stopping_early(criterion::Patience, history) =  _UP(criterion.n, history)
