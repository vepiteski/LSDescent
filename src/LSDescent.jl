module LSDescent

export bfgs_Stop, bfgs_StopLS, L_bfgs_StopLS, Newton_StopLS, Newton_Stop, Newton_Spectral

using LinearAlgebra


using NLPModels
using OneDmin

using LinearOperators
using Logging
using SolverCore   # Pour avoir les utilitaires d'affichage log_header et log_row

using Stopping

include("Newton/NewtonStopLS.jl")
include("Newton/NewtonStop.jl")
include("Newton/NewtonSolver.jl")
include("BFGS/bfgsStop.jl")
include("BFGS/bfgsStopLS.jl")
include("BFGS/FormuleN2.jl")



end # module
