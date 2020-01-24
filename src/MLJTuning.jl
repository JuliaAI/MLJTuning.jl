module MLJTuning


## METHOD EXPORT

# defined in tuned_models.jl:
export Grid, TunedModel

# defined in strategies/:
export Explicit

# defined in learning_curves.jl:
export learning_curve!, learning_curve


## METHOD IMPORT

import MLJBase
using MLJBase
using RecipesBase
using Distributed
import ComputationalResources: CPU1, CPUProcesses,
    CPUThreads, AbstractResource
using Random

## INCLUDE FILES

include("utilities.jl")
include("tuning_strategy_interface.jl")
include("tuned_models.jl")
include("ranges.jl")
include("strategies/explicit.jl")
include("strategies/grid.jl")
include("plotrecipes.jl")
include("learning_curves.jl")

end

