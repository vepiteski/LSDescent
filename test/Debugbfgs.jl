

scaling = true
maxiter = 1450


#
# Plain implementation, no Stopping, simplified Armijo (Wolfe) linesearch
#

using Logging

logger = Logging.ConsoleLogger(stderr,Logging.Warn)
Logging.with_logger(logger) do # execute twice to eliminate compilation time randomness
    # comment the definition of B₀ and in the call to have "regular" BFGS
    # use B₀ to hage L-BFGS
    # B₀ =  InverseLBFGSOperator(Float64, n, mem, scaling=scaling)
    
    @time  iter, fopt, gopt = bfgs(nlp, maxiter = maxiter, scaling = scaling) # , B₀ = B₀);
    @show iter, (gopt), fopt
end

@show nlp.counters

#
# Stopping implementation, simplified Armijo (Wolfe) linesearch
#

using Stopping

println("\n bfgs Stopping,  ")
stp = NLPStopping(nlp, NLPAtX(nlp.meta.x0)  )
stp.meta.optimality_check = unconstrained_check
stp.meta.max_iter = maxiter
stp.meta.rtol = 0  
using LSDescent

let stp = stp; #nlp = nlp
    logger = Logging.ConsoleLogger(stderr,Logging.Info)
    Logging.with_logger(logger) do # execute twice to eliminate compilation time randomness
        reset!(nlp)
        # comment the definition of B₀ and in the call to have "regular" BFGS
        # use B₀ to have L-BFGS
        #B₀ =  InverseLBFGSOperator(Float64, n, mem, scaling=scaling)
        
        @time  stp, B = bfgs_Stop(nlp, scaling = scaling, stp = stp) #, B₀ = B₀);    
        @show stp.meta.nb_of_stop, norm(stp.current_state.gx), stp.current_state.fx
    end
end

@show nlp.counters


#
# Stopping implementation, line search by 1D enhanced interval reduction
#

using OneDmin

println("\n bfgs Stopping line search,  ")

let stp = stp; #nlp = nlp
logger = Logging.ConsoleLogger(stderr,Logging.Warn)
    Logging.with_logger(logger) do # execute twice to eliminate compilation time randomness
        reset!(nlp)
        reinit!(stp)
        # comment the definition of B₀ and in the call to have "regular" BFGS
        # use B₀ to have L-BFGS
        #B₀ =  InverseLBFGSOperator(Float64, n, mem, scaling=scaling)
        lslog = Logging.ConsoleLogger(stderr,Logging.Debug)

        @time  stp, B = bfgs_StopLS(nlp, stp = stp, scaling = scaling, # B₀ = B₀,
                                              # line search options below
                                              pick_in = pick_ins2N,
                                              best = false,
                                              strongWolfe = false
                                              #, LS_logger = lslog
                                              )
        @show stp.meta.nb_of_stop, norm(stp.current_state.gx), stp.current_state.fx
    end
end

@show nlp.counters

;
