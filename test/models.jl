# If adding models from MLJModels for testing purposes, then do the
# following in the interface file (eg, DecisionTree.jl):

# - change `import ..DecisionTree` to `import DecisionTree`
# - remove wrapping as module

module Models

using MLJBase
import MLJModelInterface: @mlj_model, metadata_model, metadata_pkg
import MLJModelInterface

include("models/Constant.jl")
include("models/DecisionTree.jl")
include("models/NearestNeighbors.jl")
include("models/MultivariateStats.jl")
include("models/Transformers.jl")
include("models/foobarmodel.jl")
include("models/simple_composite_model.jl")
include("models/ensembles.jl")

end
