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

# see this julia issue: https://github.com/JuliaLang/julia/issues/34513
# if VERSION < v"1.2"
#     @testset "learning curves" begin
#         @test_broken include("learning_curves.jl")
#     end
# else
#     @test "learning curves" begin
#         @test_broken include("learning_curves.jl")
#     end
# end

# @testset "julia bug" begin
#     @test include("julia_bug.jl")
# end

