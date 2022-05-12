using Pkg
Pkg.activate(".")

# Ce script utilise plutôt des fichiers locaux...
#using LSDescent

using NLPModels, JuMP,  NLPModelsJuMP
using SolverCore, Logging
using LinearAlgebra
using LinearOperators


function test_algo(algo::Function, nlp, B, maxiter )
    reset!(nlp)

    #logger = Logging.ConsoleLogger(stderr,Logging.Warn)
    logger = Logging.NullLogger()
    Logging.with_logger(logger) do
        iter, f, normg, B, x = algo(nlp, B₀ = B, maxiter = maxiter, Lp = Inf)
        @show iter, f, normg, B.nprod
        @show norm(grad(nlp, x), Inf)
    end

end

function test_algo_stp(algo::Function, nlp, stp, B )
    reset!(nlp)

    #logger = Logging.ConsoleLogger(stderr,Logging.Warn)
    logger = Logging.NullLogger()
    Logging.with_logger(logger) do
        stp = algo(nlp, stp=stp,  B₀ = B)
        iter, f, normg =  stp.meta.nb_of_stop, stp.current_state.fx, stp.current_state.current_score
        @show iter, f, normg, stp.stopping_user_struct["BFGS"].nprod, status(stp, list=true)
        @show norm(grad(nlp, stp.current_state.x), Inf)
    end

end

function test_wrapper_stp(algo::Function, nlp, stp; kwargs... )
    reset!(nlp)

    #logger = Logging.ConsoleLogger(stderr,Logging.Warn)
    logger = Logging.NullLogger()
    Logging.with_logger(logger) do
        stp = algo(nlp, stp=stp; kwargs...)
        iter, f, normg =  stp.meta.nb_of_stop, stp.current_state.fx, stp.current_state.current_score
        @show iter, f, normg,stp.meta.optimal
    end

end




# test all solvers with a well known test function
using OptimizationProblems


#include("woods.jl")
#include("genrose.jl")
#nlp = MathOptNLPModel(PureJuMP.dixmaank(40), name="dixmaank")
#nlp = MathOptNLPModel(PureJuMP.dixmaang(100), name="dixmaang")
nlp = MathOptNLPModel(PureJuMP.srosenbr(80), name="srosenbr")
#nlp = MathOptNLPModel(PureJuMP.woods(80), name="woods")
#nlp = MathOptNLPModel(PureJuMP.genrose(80), name="genrose")


n = nlp.meta.nvar

include("Type.jl")
include("TypeCompact.jl")
include("AcceptAll.jl")

maxiter = 1600
scaling = true
#scaling = false

println()
using Stopping
include("bfgsStop.jl")

using OneDmin
include("bfgsStopLS.jl")

stp = NLPStopping(nlp, NLPAtX(nlp.meta.x0)  )
stp.meta.max_iter = maxiter


println()
@info "Version encapsulée opérateur, Cholesky"
reinit!(stp)

let scaling = scaling, n=n
    include("TypeChol.jl")
    include("FormuleChOp.jl")

    test_wrapper_stp(Ch_bfgs_Stop, nlp, stp; scaling = scaling )
    @show nlp.counters

    B = ChBFGSOperator(Float64, n; scaling = scaling);
    reset!(nlp)
    reinit!(stp)
    test_algo_stp(bfgs_StopLS, nlp, stp, B )
    @show nlp.counters
end
#



println()
@info "Version encapsulée opérateur, formule O(n^2), stp LS"
stp = NLPStopping(nlp, NLPAtX(nlp.meta.x0)  )
stp.meta.max_iter = maxiter
let scaling = scaling, n=n
    include("FormuleN2Op.jl")
    test_wrapper_stp(M_bfgs_Stop, nlp, stp; scaling = scaling )
    @show nlp.counters

    B = InverseBFGSOperator(Float64, n; scaling = scaling)
    reinit!(stp)

    test_algo_stp(bfgs_StopLS, nlp, stp, B )
    @show nlp.counters
end



println()
@info "Version L-BFGS"

reinit!(stp)

mem = 500
test_wrapper_stp(L_bfgs_Stop, nlp, stp; mem = mem, scaling = scaling )
@show nlp.counters

B = InverseLBFGSOperator(Float64, n, mem=mem; scaling = scaling)
reinit!(stp)

test_algo_stp(bfgs_StopLS, nlp, stp, B)
@show nlp.counters


println()
@info "Version Compact L-BFGS"

reinit!(stp)

test_wrapper_stp(C_bfgs_Stop, nlp, stp; mem = mem, scaling = scaling )
@show nlp.counters

B = CompactInverseBFGSOperator(Float64, n, mem=mem; scaling = scaling)
reinit!(stp)

test_algo_stp(bfgs_StopLS, nlp, stp, B)
@show nlp.counters


using LBFGSB
include("wrapper.jl")

#
println()
@info "Version L-BFGS-B Fortran"

NLPlbfgsbS = L_BFGS_B(1024, 600)
stp2 = NLPStopping(nlp, NLPAtX(nlp.meta.x0)  )
stp2.meta.max_iter = maxiter
reset!(nlp)
stp2 = NLPlbfgsbS(nlp, stp = stp2, mem = mem)#, iprint = 1)
iter, f, normg =  stp2.meta.nb_of_stop, stp2.current_state.fx, stp2.current_state.current_score
@show iter, f, normg,stp2.meta.optimal
@show nlp.counters
;
