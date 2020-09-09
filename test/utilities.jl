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

true

