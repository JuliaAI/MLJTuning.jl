module TestRanges

using Test
using MLJBase
using MLJTuning
using Random
import Distributions
const Dist = Distributions

# `in` for MLJType is overloaded to be `===` based. For purposed of
# testing here, we need `==` based:
function _in(x, itr)::Union{Bool,Missing}
    for y in itr
        ismissing(y) && return missing
        y == x && return true
    end
    return false
end
_issubset(itr1, itr2) = all(_in(x, itr2) for x in itr1)

@testset "boundedness traits" begin
    r1 = range(Float64, :K, lower=1, upper=10)
    r2 = range(Float64, :K, lower=-1, upper=Inf, origin=1, unit=1)
    r3 = range(Float64, :K, lower=0, upper=Inf, origin=1, unit=1)
    r4 = range(Float64, :K, lower=-Inf, upper=1, origin=0, unit=1)
    r5 = range(Float64, :K, lower=-Inf, upper=Inf, origin=1, unit=1)
    @test MLJTuning.boundedness(r1) == MLJTuning.Bounded
    @test MLJTuning.boundedness(r2) == MLJTuning.Other
    @test MLJTuning.boundedness(r3) == MLJTuning.PositiveUnbounded
    @test MLJTuning.boundedness(r4) == MLJTuning.Other
    @test MLJTuning.boundedness(r5) == MLJTuning.Other
end

mutable struct DummyModel <: Deterministic
    lambda::Float64
    metric::Float64
    kernel::Char
end

dummy_model = DummyModel(4, 9.5, 'k')

mutable struct SuperModel <: Deterministic
    K::Int64
    model1::DummyModel
    model2::DummyModel
end

dummy_model = DummyModel(1.2, 9.5, 'k')
super_model = SuperModel(4, dummy_model, deepcopy(dummy_model))

r1 = range(super_model, :(model1.kernel), values=['c', 'd'])
r2 = range(super_model, :K, lower=1, upper=10, scale=:log10)

@testset "models from cartesian range and resolutions" begin

    # with method:
    m1 = MLJTuning.grid(super_model, [r1, r2], [nothing, 7])
    m1r = MLJTuning.grid(MersenneTwister(123), super_model, [r1, r2],
                         [nothing, 7])

    # generate all models by hand:
    models1 = [SuperModel(1, DummyModel(1.2, 9.5, 'c'), dummy_model),
               SuperModel(1, DummyModel(1.2, 9.5, 'd'), dummy_model),
               SuperModel(2, DummyModel(1.2, 9.5, 'c'), dummy_model),
               SuperModel(2, DummyModel(1.2, 9.5, 'd'), dummy_model),
               SuperModel(3, DummyModel(1.2, 9.5, 'c'), dummy_model),
               SuperModel(3, DummyModel(1.2, 9.5, 'd'), dummy_model),
               SuperModel(5, DummyModel(1.2, 9.5, 'c'), dummy_model),
               SuperModel(5, DummyModel(1.2, 9.5, 'd'), dummy_model),
               SuperModel(7, DummyModel(1.2, 9.5, 'c'), dummy_model),
               SuperModel(7, DummyModel(1.2, 9.5, 'd'), dummy_model),
               SuperModel(10, DummyModel(1.2, 9.5, 'c'), dummy_model),
               SuperModel(10, DummyModel(1.2, 9.5, 'd'), dummy_model)]

    @test _issubset(models1, m1) && _issubset(m1, models1)
    @test m1r != models1
    @test _issubset(models1, m1r) && _issubset(m1, models1)

    # with method:
    m2 = MLJTuning.grid(super_model, [r1, r2], [1, 7])

    # generate all models by hand:
    models2 = [SuperModel(1, DummyModel(1.2, 9.5, 'c'), dummy_model),
               SuperModel(2, DummyModel(1.2, 9.5, 'c'), dummy_model),
               SuperModel(3, DummyModel(1.2, 9.5, 'c'), dummy_model),
               SuperModel(5, DummyModel(1.2, 9.5, 'c'), dummy_model),
               SuperModel(7, DummyModel(1.2, 9.5, 'c'), dummy_model),
               SuperModel(10, DummyModel(1.2, 9.5, 'c'), dummy_model)]

    @test _issubset(models2, m2) && _issubset(m2, models2)

end

@testset "processing user specification of range in Grid" begin
    r1 = range(Int, :h1, lower=1, upper=10)
    r2 = range(Int, :h2, lower=20, upper=30)
    s = range(Char, :j1, values = ['x', 'y'])
    @test_throws ArgumentError MLJTuning.process_grid_range("junk", 42, 1)
    @test(@test_logs((:warn, r"Ignoring"),
                     MLJTuning.process_grid_range((s, 3), 42, 1)) ==
          ((s, ), (2, )))
    @test MLJTuning.process_grid_range(r1, 42, 1) == ((r1, ), (42, ))
    @test MLJTuning.process_grid_range((r1, 3), 42, 1) == ((r1, ), (3, ))
    @test MLJTuning.process_grid_range(s, 42, 1) == ((s, ), (2,))
    @test MLJTuning.process_grid_range([(r1, 3), r2, s], 42, 1) ==
        ((r1, r2, s), (3, 42, 2))
end

struct MySampler end
Base.rand(rng::AbstractRNG, ::MySampler) = rand(rng)

@testset "processing user specification of range in RandomSearch" begin
    r1 = range(Int, :h1, lower=1, upper=10, scale=exp)
    r2 = range(Int, :h2, lower=5, upper=Inf, origin=10, unit=5)
    r3 = range(Char, :j1, values = ['x', 'y'])
    s = MySampler()

    @test_throws(ArgumentError,
                 MLJTuning.process_random_range("junk",
                                                Dist.Uniform,
                                                Dist.Gamma,
                                                Dist.Cauchy))
    @test_throws(ArgumentError,
                 MLJTuning.process_random_range((r1, "junk"),
                                                Dist.Uniform,
                                                Dist.Gamma,
                                                Dist.Cauchy))
    @test_throws(ArgumentError,
                 MLJTuning.process_random_range((r3, "junk"),
                                                Dist.Uniform,
                                                Dist.Gamma,
                                                Dist.Cauchy))
    @test_throws(ArgumentError,
                 MLJTuning.process_random_range(("junk", s),
                                                Dist.Uniform,
                                                Dist.Gamma,
                                                Dist.Cauchy))

    # unpaired numeric range:
    pp = MLJTuning.process_random_range(r1,
                                         Dist.Uniform, # bounded
                                         Dist.Gamma,   # positive_unbounded
                                         Dist.Cauchy)  # other
    @test pp isa Tuple{Tuple{Symbol,MLJBase.NumericSampler}}
    p = first(pp)
    @test first(p) == :h1
    s = last(p)
    @test s.scale == r1.scale
    @test s.distribution == Dist.Uniform(1.0, 10.0)

    # unpaired nominal range:
    p = MLJTuning.process_random_range(r3,
                                       Dist.Uniform,
                                       Dist.Gamma,
                                       Dist.Cauchy)  |> first
    @test first(p) == :j1
    s = last(p)
    @test s.values == r3.values
    @test s.distribution.p == [0.5, 0.5]
    @test s.distribution.support == 1:2

    # (numeric range, distribution instance):
    p = MLJTuning.process_random_range((r2, Dist.Poisson(3)),
                                       Dist.Uniform,
                                       Dist.Gamma,
                                       Dist.Cauchy) |> first
    @test first(p) == :h2
    s = last(p)
    @test s.scale == r2.scale
    @test s.distribution == Dist.truncated(Dist.Poisson(3.0), 5.0, Inf)

    # (numeric range, distribution type):
    p = MLJTuning.process_random_range((r2, Dist.Poisson),
                                       Dist.Uniform,
                                       Dist.Gamma,
                                       Dist.Cauchy) |> first
    s = last(p)
    @test s.distribution == Dist.truncated(Poisson(r2.origin), 5.0, Inf)


end
true
