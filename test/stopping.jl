module TestStopping

using Test
using MLJTuning
using MLJBase
using ..Models
using Dates

_measures = [MLJBase.LogLoss(), MLJBase.BrierScore()]

dummy_entry(loss) = entry = (model = "Junk",
                             measure = _measures,
                             measurement = [loss, 6],
                             per_fold = [1.0, 2.0])

history = dummy_entry.([10, 8, 9, 10, 11, 12, 12, 13, 14, 15, 16, 17, 16])

stopping_time(criterion, history) = findfirst(eachindex(history)) do t
    MLJTuning.stopping_early(criterion, history[1:t])
end

X, y = @load_boston;

@testset "_loss" begin
    entry = (model = "Junk",
             measure = _measures,
             measurement = [5, 6],
             per_fold = [1.0, 2.0])
    @test MLJTuning._loss(entry) == 5
    entry = (model = "Junk",
             measure = reverse(_measures),
             measurement = reverse([5, 6]),
             per_fold = [1.0, 2.0])
    @test MLJTuning._loss(entry) ==  -6
end


@testset "Never" begin
    @test stopping_time(Never(), history) === nothing
end

@testset "TimeLimit" begin
    @test_throws ArgumentError TimeLimit(t=0)
    @test TimeLimit(1).t == Millisecond(3_600_000)
    @test TimeLimit(t=Day(2)).t == Millisecond(48*3_600_000)
end

@testset "GeneralizationLoss" begin
    # constructor:
    @test_throws ArgumentError GeneralizationLoss(alpha=0)
    @test GeneralizationLoss(alpha=1).alpha === 1.0

    # _GL:
    by_hand = [0, 0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 8] .* 100 ./8
    by_code = map(eachindex(history)) do t
        MLJTuning._GL(history[1:t])
    end

    # stopping times:
    @test by_hand ≈ by_code
    @test stopping_time(GeneralizationLoss(alpha=12), history) == 3
    @test stopping_time(GeneralizationLoss(alpha=20), history) == 4
    @test stopping_time(GeneralizationLoss(alpha=40), history) == 6
    @test stopping_time(GeneralizationLoss(alpha=90), history) == 11
    @test stopping_time(GeneralizationLoss(alpha=110), history) == 12
    @test stopping_time(GeneralizationLoss(alpha=1000), history) === nothing
end

@testset "_UP" begin
    @test all(t -> !MLJTuning._UP(6, history[1:t]), eachindex(history))
    @test all(n -> !MLJTuning._UP(n, history[1:1]), eachindex(history))
    @test !MLJTuning._UP(2, history[1:3])
    @test MLJTuning._UP(2, history[1:4])
    @test MLJTuning._UP(4, history[1:11])
end

@testset "Patience" begin
    @test_throws ArgumentError Patience(n=0)
    @test stopping_time(Patience(n=6), history) === nothing
    @test stopping_time(Patience(n=5), history) == 12
    @test stopping_time(Patience(n=4), history) == 6
    @test stopping_time(Patience(n=3), history) == 5
    @test stopping_time(Patience(n=2), history) == 4
    @test stopping_time(Patience(n=1), history) == 3
end


## INTEGRATION TESTS

model = KNNRegressor()
r = range(model, :K, lower=1, upper=10)

# next commented-out section shows mean absolute error for model
# on 30% holdout set, as a function of the parameter `K` (all
# other parameters taking on default values):

# _, _, k_values, errors =
#     learning_curve(model, X, y, range=r, measure=mae, resolution=100)
# zip(k_values, errors) |> collect
# 10-element Array{Tuple{Int64,Float64},1}:
#  (1, 5.625657894736842)  #       GL=0
#  (2, 5.383552631578948)  # down  GL=0
#  (3, 5.525877192982458)  # up    GL=2.6
#  (4, 5.441118421052633)  # down  GL=1.1
#  (5, 5.382105263157899)  # down  GL=0
#  (6, 5.480921052631575)  # up    GL=1.9
#  (7, 5.5624060150375945) # up    GL=3.4
#  (8, 5.737499999999999)  # up    GL=6.60
#  (9, 5.745833333333332)  # up    GL=6.76
#  (10, 5.7358552631578945) # down GL=6.57

function stopping_time(criterion)
    tuned_model = TunedModel(model=model,
                             tuning=Grid(resolution=100, shuffle=false),
                             stopping=criterion,
                             range=r,
                             measure=mae)
    mach = machine(tuned_model, X, y)
    fit!(mach, verbosity=0)
    report(mach).history |> length
end

@testset "TimeLimit - integration test" begin
    model = FooBarRegressor() # baby ridge regressor
    r = range(model, :lambda, lower=0.0001, upper = 10)
    tuned_model = TunedModel(model=model,
                             tuning=Grid(shuffle=false, resolution=1e5),
                             stopping=TimeLimit(t=Second(1)),
                             range=r,
                             measure=mae)
    mach = machine(tuned_model, X, y)
    @test_logs (:info, r"Stopping early") fit!(mach, verbosity=0, force=true)
    tuned_model.stopping = TimeLimit(t=Second(5))
    t = @elapsed fit!(mach, verbosity=-1, force=true)
    @test abs(t - 5)/5 < 0.5 # wait within 50% of requested
end

@testset "GeneralizationLoss - integration test" begin
    @test_logs((:info, r"Stopping early after 3"),
               @test stopping_time(GeneralizationLoss(1)) == 3)
    @test stopping_time(GeneralizationLoss(1)) == 3
    @test stopping_time(GeneralizationLoss(2)) == 3
    @test stopping_time(GeneralizationLoss(3)) == 7
    @test stopping_time(GeneralizationLoss(6)) == 8
    @test stopping_time(GeneralizationLoss(6.65)) == 9
    @test stopping_time(GeneralizationLoss(6.8)) == 10
end

@testset "Patience - integration test" begin
    @test_logs((:info, r"Stopping early after 3"),
               @test stopping_time(Patience(1)) == 3)
    @test stopping_time(Patience(2)) == 7
    @test stopping_time(Patience(3)) == 8
    @test stopping_time(Patience(4)) == 9
    @test stopping_time(Patience(5)) == 10
    @test stopping_time(Patience(6)) == 10
end

end

true
