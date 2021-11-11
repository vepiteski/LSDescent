using LSDescent

using Test
using NLPModels
using JuMP


# test all solvers with the well known Woods test function
include("woods.jl")
nlp = MathProgNLPModel(woods(), name="woods")


nbsolver = 0
@printf("Testing Stopping Newton solvers\n\n")

using Stopping

println("\n Newton-spectral Stopping,  ")
stp = NLPStopping(nlp, NLPAtX(nlp.meta.x0)  )
stp.meta.optimality_check = unconstrained_check
stp.meta.max_iter = maxiter
stp.meta.rtol = 0

#include("Newton/NewtonStop.jl")

stp = Newton_Stop(nlp, stp = stp);
@show stp.meta.nb_of_stop, norm(stp.current_state.gx), stp.current_state.fx
@show nlp.counters

