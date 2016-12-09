module LSDescentMethods

using NLPModels
using Optimize
using LinearOperators


AbstractLineFunction = Union{C1LineFunction,C2LineFunction}

include("lbfgs.jl")
include("steepest.jl")
include("CG_PR.jl")
include("NewtonSpectralAbs.jl")
include("armijo_wolfe.jl")

using LineSearch



end # module
