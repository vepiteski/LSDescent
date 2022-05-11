using Pkg
Pkg.activate(".")

using NLPModels, JuMP,  NLPModelsJuMP

using SolverCore, Logging

using LinearOperators

using LinearAlgebra


function test_algo_stp(algo::Function, nlp, stp, B )
    reset!(nlp)

    logger = Logging.ConsoleLogger(stderr,Logging.Warn)
    Logging.with_logger(logger) do
        stp = algo(nlp, stp=stp, B₀=B)
        iter, f, normg =  stp.meta.nb_of_stop, stp.current_state.fx, stp.current_state.current_score
        @show iter, f, normg
    end

end

function test_wrapper_stp(algo::Function, nlp, stp; kwargs... )
    reset!(nlp)

    logger = Logging.ConsoleLogger(stderr,Logging.Warn)
    Logging.with_logger(logger) do
        stp = algo(nlp, stp=stp; kwargs...)
        iter, f, normg =  stp.meta.nb_of_stop, stp.current_state.fx, stp.current_state.current_score
        @show iter, f, normg
    end

end


# test all solvers with the well known Woods test function
using OptimizationProblems


#include("woods.jl")
#include("genrose.jl")
#nlp = MathOptNLPModel(PureJuMP.dixmaank(40), name="dixmaank")
#nlp = MathOptNLPModel(PureJuMP.dixmaang(100), name="dixmaang")
#nlp = MathOptNLPModel(PureJuMP.srosenbr(10), name="srosenbr")
nlp = MathOptNLPModel(PureJuMP.woods(8), name="woods")
#nlp = MathOptNLPModel(PureJuMP.genrose(100), name="genrose")
n = nlp.meta.nvar


maxiter = 1600
scaling = false

include("Type.jl")


@info "Version encapsulée opérateur, formule O(n^2), stp"
using Stopping
include("bfgsStop.jl")

stp = NLPStopping(nlp,
                  NLPAtX(nlp.meta.x0)  )
stp.meta.max_iter = maxiter


B = InverseBFGSOperator(Float64, n; scaling = scaling)
test_algo_stp(bfgs_Stop, nlp, stp, B )


@info "Version encapsulée opérateur, formule O(n^2), stp LS"
using OneDmin
include("bfgsStopLS.jl")

reinit!(stp)
test_wrapper_stp(M_bfgs_StopLS, nlp, stp; scaling = scaling )

reinit!(stp)
B = InverseBFGSOperator(Float64, n; scaling = scaling)
test_algo_stp(bfgs_StopLS, nlp, stp, B )


@info "Version L-BFGS"

reinit!(stp)
test_wrapper_stp(L_bfgs_Stop, nlp, stp; scaling = scaling )

reinit!(stp)
B = InverseLBFGSOperator(Float64, n, mem=160; scaling = scaling)
test_algo_stp(bfgs_Stop, nlp, stp, B)


@info "Version Compact L-BFGS"

reinit!(stp)
test_wrapper_stp(C_bfgs_Stop, nlp, stp; scaling = scaling )

reinit!(stp)
B = CompactInverseBFGSOperator(Float64, n, mem=160; scaling = scaling)
test_algo_stp(bfgs_Stop, nlp, stp, B)

;

