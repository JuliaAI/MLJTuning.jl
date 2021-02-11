module MLJTuning


## METHOD EXPORT

# defined in stopping.jl:
export StoppingCriterion, Never, TimeLimit, Patience

# defined in tuned_models.jl:
export TunedModel

# defined in strategies/:
export Explicit, Grid, RandomSearch, LatinHypercube

# defined in selection_heuristics/:
export NaiveSelection

# defined in learning_curves.jl:
export learning_curve!, learning_curve


## METHOD IMPORT

import MLJBase
using MLJBase
import MLJBase: Bounded, Unbounded, DoublyUnbounded,
    LeftUnbounded, RightUnbounded, _process_accel_settings, chunks
using RecipesBase
using Distributed
import Distributions
import ComputationalResources: CPU1, CPUProcesses,
    CPUThreads, AbstractResource
using Random
using ProgressMeter
using LatinHypercubeSampling
using Dates


## CONSTANTS

const DEFAULT_N = 10 # for when `default_n` is not implemented

## INCLUDE FILES

include("utilities.jl")
include("stopping.jl")
include("tuning_strategy_interface.jl")
include("selection_heuristics.jl")
include("strategies/explicit.jl")
include("strategies/grid.jl")
include("strategies/random_search.jl")
include("strategies/latin_hypercube.jl")
include("tuned_models.jl")
include("range_methods.jl")
include("plotrecipes.jl")
include("learning_curves.jl")

end
