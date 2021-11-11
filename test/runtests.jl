using LSDescent

using Test
using NLPModels, JuMP,  NLPModelsJuMP

using SolverTools

using LinearAlgebra

# test all solvers with the well known Woods test function
include("woods.jl")
nlp = MathOptNLPModel(woods(), name="woods")


nbsolver = 0

maxiter = 100
println("Testing bfgs solvers\n\n")
include("Debugbfgs.jl")


println("Testing Newton solvers\n\n")
include("DebugNewton.jl")
