using Pkg
Pkg.activate("wo3", shared=true)
using MLJBase
using MLJTuning
include("models.jl")
using .Models
using Random
using Test

# for debugging (rng's have verbose shows):
Base.show(io::IO, ::AbstractRNG)  = print("<RNG>")
Base.show(io::IO, ::MIME"text/plain", ::AbstractRNG)  = print("<RNG>")


N = 30
x1 = rand(N);
x2 = rand(N);
x3 = rand(N);
X = (x1=x1, x2=x2, x3=x3);
y = 2*x1 .+ 5*x2 .- 3*x3 .+ 0.4*rand(N);

m(K) = KNNRegressor(K=K)
r = [m(K) for K in 2:13]

@testset "constructor" begin
    @test_throws ErrorException TunedModel(model=first(r), tuning=Explicit(),
                                           measure=rms)
    @test_throws ErrorException TunedModel(tuning=Explicit(),
                                           range=r, measure=rms)
    @test_throws ErrorException TunedModel(model=42, tuning=Explicit(),
                                           range=r, measure=rms)
    TunedModel(model=first(r), tuning=Explicit(), range=r)
end





# old julia_bug.jl:
x1 = rand(100);
x2 = rand(100);
x3 = rand(100);
X = (x1=x1, x2=x2, x3=x3);
y = 2*x1 .+ 5*x2 .- 3*x3 .+ 0.2*rand(100);


atom = FooBarRegressor()
ensemble = EnsembleModel(atom=atom, n=50)
mach = machine(ensemble, X, y)
r_n = range(ensemble, :n, lower=10, upper=100)

curves = learning_curve!(mach; range=r_n, resolution=7,
                         rngs = 3,
                         rng_name=:rng)


true
