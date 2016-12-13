module LSDescentMethods

using NLPModels
using Optimize
using LinearOperators


AbstractLineFunction = Union{C1LineFunction,C2LineFunction}

include("lbfgs.jl")
include("steepest.jl")
include("CG_FR.jl")
include("CG_PR.jl")
include("CG_HS.jl")
include("CG_HZ.jl")
include("NewtonSpectralAbs.jl")
include("NewtonLDLtAbs.jl")
using HSL
include("NewtonMA57Abs.jl")
include("armijo_wolfe.jl")

using LineSearch



end # module
