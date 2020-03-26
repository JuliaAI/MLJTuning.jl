module MLJTuning


## METHOD EXPORT

# defined in tuned_models.jl:
export TunedModel

# defined in strategies/:
export Explicit, Grid

# defined in learning_curves.jl:
export learning_curve!, learning_curve


## METHOD IMPORT

import MLJBase
using MLJBase
import MLJBase: Bounded, Unbounded, DoublyUnbounded,
    LeftUnbounded, RightUnbounded
using RecipesBase
using Distributed
import Distributions
import ComputationalResources: CPU1, CPUProcesses,
    CPUThreads, AbstractResource
using Random


## CONSTANTS

const DEFAULT_N = 10 # for when `default_n` is not implemented


## INCLUDE FILES

include("utilities.jl")
include("tuning_strategy_interface.jl")
include("tuned_models.jl")
include("range_methods.jl")
include("strategies/explicit.jl")
include("strategies/grid.jl")
include("plotrecipes.jl")
include("learning_curves.jl")

end

