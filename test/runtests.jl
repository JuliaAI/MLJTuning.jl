using Distributed
addprocs(2)

using Test
using MLJTuning
using MLJBase

# Display Number of processes and if necessary number
# of Threads
@info "nworkers: $(nworkers())"
@static if VERSION >= v"1.3.0-DEV.573"
    @info "nthreads: $(Threads.nthreads())"
else
    @info "Running julia $(VERSION). Multithreading tests excluded. "
end

include("test_utilities.jl")

print("Loading some models for testing...")
# load `Models` module containing models implementations for testing:
include("models.jl") #precompile first in master node
print("\r                                           \r")
@everywhere include("models.jl")

@testset "utilities" begin
    @test include("utilities.jl")
end

@testset "selection heuristics" begin
    include("selection_heuristics.jl")
end

@testset "tuned_models.jl" begin
    @test include("tuned_models.jl")
end

@testset "range_methods" begin
    @test include("range_methods.jl")
end

@testset "grid" begin
    @test include("strategies/grid.jl")
end

@testset "random search" begin
    @test include("strategies/random_search.jl")
end

@testset "Latin hypercube" begin
    @test include("strategies/latin_hypercube.jl")
end

@testset "learning curves" begin
        @test include("learning_curves.jl")
end

# @testset "julia bug" begin
#     @test include("julia_bug.jl")
# end

