using MLJFlux, MLJ, Flux

X = randn(100, 2)
Y = X * rand(2, 2) .+ 0.1 * randn.()
XT = MLJ.table(X, names = [:x1, :x2])
YT = MLJ.table(Y, names = [:y1, :y2])
act = tanh
nn = Chain(
  Dense(2, 5, act),
  Dense(5, 5, act),
  Dense(5, 5, act),
  Dense(5, 2, identity),
)
builder = MLJFlux.@builder nn

function multi_target(loss)
  (x1, x2) -> sum(map(x1, x2) do _x1, _x2
    loss(_x1, _x2)
  end)
end
loss = multi_target(l2)
model = MLJFlux.MultitargetNeuralNetworkRegressor(builder = builder; epochs = 10, loss)
r = (MLJ.range(model, :lambda, lower=1e-6, upper=1.0, scale=:linear), 10)
tuning = MLJ.Grid(shuffle = true)
tuned_model = MLJ.TunedModel(
  model;
  tuning,
  resampling = MLJ.CV(nfolds = 5),
  range = [r],
  measure = loss,
  n = 10,
  check_measure = false,
)
mach = MLJ.machine(tuned_model, XT, YT)
MLJ.fit!(mach, verbosity=1)
