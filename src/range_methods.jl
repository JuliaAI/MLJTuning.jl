## BOUNDEDNESS TRAIT

# For random search and perhaps elsewhere, we need a variation on the
# built-in boundedness notions:
abstract type PositiveUnbounded <: Unbounded end
abstract type Other <: Unbounded end

boundedness(::NumericRange{<:Any,<:Bounded}) = Bounded
boundedness(::NumericRange{<:Any,<:LeftUnbounded}) = Other
boundedness(::NumericRange{<:Any,<:DoublyUnbounded}) = Other
function boundedness(r::NumericRange{<:Any,<:RightUnbounded})
    if r.lower >= 0
        return PositiveUnbounded
    end
    return Other
end


## PRE-PROCESSING OF USER-SPECIFIED CARTESIAN RANGE OBJECTS

"""
    MLJTuning.grid([rng, ] prototype, ranges, resolutions)

Given an iterable `ranges` of `ParamRange` objects, and an iterable
`resolutions` of the same length, return a vector of models generated
by cloning and mutating the hyperparameters (fields) of `prototype`,
according to the Cartesian grid defined by the specifed
one-dimensional `ranges` (`ParamRange` objects) and specified
`resolutions`. A resolution of `nothing` for a `NominalRange`
indicates that all values should be used.

Specification of an `AbstractRNG` object `rng` implies shuffling of
the results. Otherwise models are ordered, with the first
hyperparameter referenced cycling fastest.

"""
grid(rng::AbstractRNG, prototype::Model, ranges, resolutions) =
    shuffle(rng, grid(prototype, ranges, resolutions))

function grid(prototype::Model, ranges, resolutions)

    iterators = broadcast(iterator, ranges, resolutions)

    A = MLJBase.unwind(iterators...)

    N = size(A, 1)
    map(1:N) do i
        clone = deepcopy(prototype)
        for k in eachindex(ranges)
            field = ranges[k].field
            recursive_setproperty!(clone, field, A[i,k])
        end
        clone
    end
end

"""
    process_grid_range(user_specified_range, resolution, verbosity)

Utility to convert a user-specified range (see [`Grid`](@ref)) into a
pair of tuples `(ranges, resolutions)`.

For example, if `r1`, `r2` are `NumericRange`s and `s` is a
NominalRange` with 5 values, then we have:

    julia> MLJTuning.process_grid_range([(r1, 3), r2, s], 42, 1) ==
                            ((r1, r2, s), (3, 42, 5))
    true

If `verbosity` > 0, then a warning is issued if a `Nominal` range is
paired with a resolution.

"""
process_grid_range(user_specified_range, args...) =
    throw(ArgumentError("Unsupported range. "))

process_grid_range(usr::Union{ParamRange,Tuple{ParamRange,Int}}, args...) =
    process_grid_range([usr, ], args...)

function process_grid_range(user_specified_range::AbstractVector,
                    resolution, verbosity)
    # r unpaired:
    stand(r) = throw(ArgumentError("Unsupported range. "))
    stand(r::NumericRange) = (r, resolution)
    stand(r::NominalRange) = (r, length(r.values))

    # (r, res):
    stand(t::Tuple{NumericRange,Integer}) = t
    function stand(t::Tuple{NominalRange,Integer})
        verbosity < 0 ||
            @warn  "Ignoring a resolution specified for a `NominalRange`. "
        return (first(t), length(first(t).values))
    end

    ret = zip(stand.(user_specified_range)...) |> collect
    return first(ret), last(ret)
end

"""
    process_random_range(user_specified_range,
                         bounded,
                         positive_unbounded,
                         other)

Utility to convert a user-specified range (see [`RandomSearch`](@ref))
into an n-tuple of `(field, sampler)` pairs.

"""
process_random_range(user_specified_range, args...) =
    throw(ArgumentError("Unsupported range #1. "))

const DIST = Distributions.Distribution

process_random_range(user_specified_range::Union{ParamRange, Tuple{Any,Any}},
                     args...) =
    process_random_range([user_specified_range, ], args...)

function process_random_range(user_specified_range::AbstractVector,
                              bounded,
                              positive_unbounded,
                              other)
    # r not paired:
    stand(r) = throw(ArgumentError("Unsupported range #2. "))
    stand(r::NumericRange) = stand(r, boundedness(r))
    stand(r::NumericRange, ::Type{<:Bounded}) = (r.field, sampler(r, bounded))
    stand(r::NumericRange, ::Type{<:Other}) = (r.field, sampler(r, other))
    stand(r::NumericRange, ::Type{<:PositiveUnbounded}) =
        (r.field, sampler(r, positive_unbounded))
    stand(r::NominalRange) = (n = length(r.values);
                              (r.field, sampler(r, fill(1/n, n))))
    # (r, d):
    stand(t::Tuple{ParamRange,Any}) = stand(t...)
    stand(r, d) = throw(ArgumentError("Unsupported range #3. "))
    stand(r::NominalRange, d::AbstractVector{Float64}) =  _stand(r, d)
    stand(r::NumericRange, d:: Union{DIST, Type{<:DIST}}) = _stand(r, d)
    _stand(r, d) = (r.field, sampler(r, d))

    # (field, s):
    stand(t::Tuple{Union{Symbol,Expr},Any}) = t

    return  Tuple(stand.(user_specified_range))

    # ret = zip(stand.(user_specified_range)...) |> collect
    # return first(ret), last(ret)
end
