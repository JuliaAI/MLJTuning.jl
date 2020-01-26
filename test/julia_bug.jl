using MLJBase
using MLJTuning
using .Models
using Random

# for debugging (rng's have verbose shows):
Base.show(io::IO, ::AbstractRNG)  = print("<RNG>")
Base.show(io::IO, ::MIME"text/plain", ::AbstractRNG)  = print("<RNG>")

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
