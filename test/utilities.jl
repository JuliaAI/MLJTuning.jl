@testset "unwind function" begin
    iterators = ([1, 2], ["a","b"], ["x", "y", "z"])
    @test MLJTuning.unwind(iterators...) ==
        [1  "a"  "x";
         2  "a"  "x";
         1  "b"  "x";
         2  "b"  "x";
         1  "a"  "y";
         2  "a"  "y";
         1  "b"  "y";
         2  "b"  "y";
         1  "a"  "z";
         2  "a"  "z";
         1  "b"  "z";
         2  "b"  "z"]
end

@test MLJTuning.delete((x=1, y=2, z=3), :x, :z) == (y=2,)

@testset "signature of measure" begin
    measures = [accuracy, confmat, misclassification_rate]
    @test MLJTuning.signature.(measures) == [-1, 0, 1]
end

@testset "trim" begin
    str = "some.long.name" # 14 characters
    @test MLJTuning.trim(str, 14) == str
    @test MLJTuning.trim(str, 13) == "…long.name" # 10 characters
    @test MLJTuning.trim(str, 12) == "…long.name"
    @test MLJTuning.trim(str, 11) == "…long.name"
    @test MLJTuning.trim(str, 10) == "…long.name"
    @test MLJTuning.trim(str, 9) == "…name"
    @test MLJTuning.trim(str, 1) == "…name" # cannot go any smaller
end

true

