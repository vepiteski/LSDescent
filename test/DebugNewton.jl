
#
# Plain implementation, no Stopping, simplified Armijo (Wolfe) linesearch
#

using Logging

logger = Logging.ConsoleLogger(stderr,Logging.Warn)
Logging.with_logger(logger) do # execute twice to eliminate compilation time randomness
    @time iter, fopt, gopt = Newton_Spectral(nlp, maxiter = maxiter)

    reset!(nlp)
    @time iter, fopt, gopt = Newton_Spectral(nlp, maxiter = maxiter)
    @show iter, (fopt), gopt
end

@show nlp.counters

#
# Stopping implementation, simplified Armijo (Wolfe) linesearch
#


using Stopping

println("\n Newton-spectral Stopping,  ")
stp = NLPStopping(nlp, NLPAtX(nlp.meta.x0)  )
stp.meta.optimality_check = unconstrained_check
stp.meta.max_iter = maxiter
stp.meta.rtol = 0

stp2 = NLPStopping(nlp2, NLPAtX(nlp.meta.x0)  )
stp2.meta.optimality_check = unconstrained_check
stp2.meta.max_iter = maxiter
stp2.meta.rtol = 0



reset!(nlp)
reinit!(stp)

include("Newton/NewtonStop.jl")

let stp = stp
    logger = Logging.ConsoleLogger(stderr,Logging.Warn)
    Logging.with_logger(logger) do # execute twice to eliminate compilation time randomness
        reset!(nlp)
        reinit!(stp)
        @time  stp = Newton_Stop(nlp, stp = stp);
        reset!(nlp)
        reinit!(stp)
        @time  stp = Newton_Stop(nlp, stp = stp);

        @show stp.meta.nb_of_stop, norm(stp.current_state.gx), stp.current_state.fx
    end
end

@show nlp.counters




#
# Stopping implementation, line search by 1D enhanced interval reduction
#

include("Newton/NewtonStopLS.jl")

using OneDmin

println("\n Newton-spectral Stopping line search, interval reduction line search ")

logger = Logging.ConsoleLogger(stderr,Logging.Warn)
Logging.with_logger(logger) do # execute twice to eliminate compilation time randomness
    reset!(nlp2)
    reinit!(stp2)
    lslog = Logging.ConsoleLogger(stderr,Logging.Debug)
    
    @time  stp3 = Newton_StopLS(nlp2, stp = stp2,
                                pick_in = pick_ins2,
                                best = false,
                                strongWolfe = true
                                )
    reset!(nlp)
    reinit!(stp)
    
    reset!(nlp2)
    reinit!(stp2)
    lslog = Logging.ConsoleLogger(stderr,Logging.Debug)
    
    @time  stp3 = Newton_StopLS(nlp2, stp = stp2,
                                pick_in = pick_ins2,
                                best = false,
                                strongWolfe = true
                                )
    reset!(nlp)
    reinit!(stp)
    
    
    @time  stp4 = Newton_StopLS(nlp, stp = stp,
                                pick_in = pick_ins2,
                                best = false,
                                strongWolfe = true
                                )
    @show stp.meta.nb_of_stop, norm(stp.current_state.gx), stp.current_state.fx
end

@show nlp.counters


;
#include("OneD/TR-US_mod.jl")

println("\n Newton-spectral Stopping line search, TR line search  ")

logger = Logging.ConsoleLogger(stderr,Logging.Warn)
    Logging.with_logger(logger) do # execute twice to eliminate compilation time randomness
        reset!(nlp2)
        reinit!(stp2)
        lslog = Logging.ConsoleLogger(stderr,Logging.Debug)

        @time  stp3 = Newton_StopLS(nlp2, stp = stp2,
                                              strongWolfe = true,
                                              LS_algo = TR1D
                                              #, LS_logger = lslog
                                              )
        reset!(nlp)
        reinit!(stp)

        reset!(nlp2)
        reinit!(stp2)
        lslog = Logging.ConsoleLogger(stderr,Logging.Info)

        @time  stp3 = Newton_StopLS(nlp2, stp = stp2,
                                              strongWolfe = true,
                                              LS_algo = TR1D
                                              #, LS_logger = lslog
                                              )
        reset!(nlp)
        reinit!(stp)


        @time  stp4 = Newton_StopLS(nlp, stp = stp,
                                              strongWolfe = false,
                                              LS_algo = TR1D
                                              #, LS_logger = lslog
                                              )
        @show stp.meta.nb_of_stop, norm(stp.current_state.gx), stp.current_state.fx
    end

@show nlp.counters


finalize(nlp)
