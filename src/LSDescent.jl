module LSDescent

export bfgs, bfgs_Stop, bfgs_StopLS
export L_bfgs, L_bfgs_Stop, L_bfgs_StopLS
export Newton_StopLS, Newton_Stop, Newton_Spectral
export CG_generic, formula_FR, formula_PR, formula_HS, formula_HZ
export CG_FR, CG_PR, CG_HS, CG_HZ

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

export Newton_G_StopLS, Newton_G_Stop
include("Newton/Newton_G_StopLS.jl")
include("Newton/Newton_G_Stop.jl")

include("Newton/HessianDense.jl")
include("Newton/HessianOp.jl")
include("Newton/HessianSparse.jl")

include("Newton/NewtonSpectralAbs.jl")

include("Newton/ldlt_symm.jl")
export ldlt_symm
include("Newton/NewtonLDLtAbs.jl")
include("Newton/NewtonCG.jl")




include("BFGS/bfgsSolver.jl")
include("BFGS/bfgsStop.jl")
include("BFGS/bfgsStopLS.jl")
include("BFGS/FormuleN2.jl")

include("ConjugateGradient/CG_generic.jl")
include("ConjugateGradient/formulae.jl")

include("ConjugateGradient/CG_FR.jl")
include("ConjugateGradient/CG_PR.jl")
include("ConjugateGradient/CG_HS.jl")
include("ConjugateGradient/CG_HZ.jl")


end # module
