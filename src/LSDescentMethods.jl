module LSDescentMethods

export ALL_solvers, STOP_solvers

using NLPModels
using Optimize

using LinearOperators

using PyPlot

using Stopping

include("armijo_wolfe.jl")

# AbstractLineFunction = Union{C1LineFunction,C2LineFunction}
ALL_solvers = Function[]
STOP_solvers = Function[]

include("lbfgs.jl")
push!(ALL_solvers,Newlbfgs)
include("steepest.jl")
#push!(ALL_solvers,steepest)
include("formulae.jl")
include("CG_generic.jl")
include("CG_FR.jl")
push!(ALL_solvers,CG_FR)
include("CG_PR.jl")
push!(ALL_solvers,CG_PR)
include("CG_HS.jl")
push!(ALL_solvers,CG_HS)
include("CG_HZ.jl")
push!(ALL_solvers,CG_HZ)
include("HessianDense.jl")
include("NewtonSpectralAbs.jl")
include("NewtonLDLtAbs.jl")
#using HSL
include("HessianSparse.jl")
#include("NewtonMA57Abs.jl")
include("HessianOp.jl")

include("cgTN.jl")
include("NewtonCG.jl")
include("Newton.jl")
push!(ALL_solvers,Newton)

include("Stopping/CG_FRS.jl")
include("Stopping/CG_genericS.jl")
include("Stopping/CG_HSS.jl")
include("Stopping/CG_HZS.jl")
include("Stopping/CG_PRS.jl")
include("Stopping/lbfgsS.jl")
include("Stopping/NewtonS.jl")
include("Stopping/steepestS.jl")

include("BFGS_Jo.jl")


push!(STOP_solvers,NewlbfgsS)
push!(STOP_solvers,CG_FRS)
push!(STOP_solvers,CG_PRS)
push!(STOP_solvers,CG_HSS)
push!(STOP_solvers,CG_HZS)

push!(STOP_solvers,NewtonS)


using LineSearch, LineSearches



end # module
