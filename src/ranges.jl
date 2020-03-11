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
    process_user_range(user_specified_range, resolution, verbosity)

Utility to convert user-specified range (see [`Grid`](@ref)) into a
pair of tuples `(ranges, resolutions)`.

For example, if `r1`, `r2` are `NumericRange`s and `s` is a
NominalRange` with 5 values, then we have:

    julia> MLJTuning.process_user_range([(r1, 3), r2, s], 42, 1) ==
                            ((r1, r2, s), (3, 42, 5))
    true

If `verbosity` > 0, then a warning is issued if a `Nominal` range is
paired with a resolution.

"""
process_user_range(user_specified_range, resolution, verbosity) =
    process_user_range([user_specified_range, ], resolution, verbosity)
function process_user_range(user_specified_range::AbstractVector,
                    resolution, verbosity)
    stand(r) = throw(ArgumentError("Unsupported range. "))
    stand(r::NumericRange) = (r, resolution)
    stand(r::NominalRange) = (r, length(r.values))
    stand(t::Tuple{NumericRange,Integer}) = t
    function stand(t::Tuple{NominalRange,Integer})
        verbosity < 0 ||
            @warn  "Ignoring a resolution specified for a `NominalRange`. "
        return (first(t), length(first(t).values))
    end

    ret = zip(stand.(user_specified_range)...) |> collect
    return first(ret), last(ret)
end
