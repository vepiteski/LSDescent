 
include("NewtonStopLS.jl")
include("NewtonStop.jl")
include("NewtonSolver.jl")

include("Newton_G_StopLS.jl")
include("Newton_G_Stop.jl")

include("HessianDense.jl")
include("HessianOp.jl")
include("HessianSparse.jl")

include("NewtonSpectralAbs.jl")

export ldlt_symm
include("NewtonLDLtAbs.jl")
include("NewtonCG.jl")

