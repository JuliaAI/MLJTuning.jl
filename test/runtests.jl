using Distributed
addprocs(2)

using Test
using MLJTuning
using MLJBase

include("test_utilities.jl")

print("Loading some models for testing...")
# load `Models` module containing models implementations for testing:
@everywhere include("models.jl")
print("\r                                           \r")

@testset "utilities" begin
    @test include("utilities.jl")
end

@testset "tuned_models.jl" begin
    @test include("tuned_models.jl")
end

@testset "ranges" begin
    @test include("ranges.jl")
end

@testset "grid" begin
    @test include("strategies/grid.jl")
end

@testset "learning curves" begin
    @test include("learning_curves.jl")
end
