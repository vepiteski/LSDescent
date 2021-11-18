maxiter = 1000

using Stopping

stp = NLPStopping(nlp, NLPAtX(nlp.meta.x0)  )

Lp = Inf

my_unconstrained_check(nlp, st; kwargs...) = unconstrained_check(nlp, st, pnorm = Lp; kwargs...)
stp.meta.optimality_check = my_unconstrained_check

#stp.meta.optimality_check = unconstrained_check
stp.meta.max_iter = maxiter
stp.meta.rtol = 0  

using OneDmin


function test_Stp(algo::Function, nlp; kwargs...) where T

    stats = @timed bidon=0
    logger = Logging.ConsoleLogger(stderr,Logging.Warn)
    Logging.with_logger(logger) do 
        stats = @timed stp = algo(nlp; kwargs...)
    end
        
    return stats, stp
end


using Test
@info log_header([:name, :time, :iter, :f, :dual], [String, Float64, Int, Float64, Float64],
                 hdr_override=Dict(:name => "Solver", :f=>"f(x)", :dual=>"‖∇f‖"))

println("\n CG   ")


stats, stp = test_Stp(CG_generic, nlp, stp=stp, scaling = false, strongWolfe = true, LS_algo=bracket_N, CG_formula=formula_HZ)

@info log_row(Any["CG_HZ", stats.time,  stp.meta.nb_of_stop, stp.current_state.fx, stp.current_state.current_score])
@test stp.current_state.current_score < 1e-6

reset!(nlp)
reinit!(stp)

stats, stp = test_Stp(CG_HZ, nlp, stp=stp, scaling = false, strongWolfe = true, LS_algo=bracket_N)

@info log_row(Any["CG_HZ wrapper", stats.time,  stp.meta.nb_of_stop, stp.current_state.fx, stp.current_state.current_score])
@test stp.current_state.current_score < 1e-6

reset!(nlp)
reinit!(stp)

stats, stp = test_Stp(CG_HS, nlp, stp=stp, scaling = false, strongWolfe = true, LS_algo=bracket_N)

@info log_row(Any["CG_HS", stats.time,  stp.meta.nb_of_stop, stp.current_state.fx, stp.current_state.current_score])
@test stp.current_state.current_score < 1e-6

reset!(nlp)
reinit!(stp)

stats, stp = test_Stp(CG_PR, nlp, stp=stp, scaling = false, strongWolfe = true, LS_algo=bracket_N)

@info log_row(Any["CG_PR", stats.time,  stp.meta.nb_of_stop, stp.current_state.fx, stp.current_state.current_score])
@test stp.current_state.current_score < 1e-6

reset!(nlp)
reinit!(stp)

stats, stp = test_Stp(CG_generic, nlp, stp=stp, scaling = true, strongWolfe = true, LS_algo=bracket_N, CG_formula=formula_FR)

@info log_row(Any["CG_FR", stats.time,  stp.meta.nb_of_stop, stp.current_state.fx, stp.current_state.current_score])
@test stp.current_state.current_score < 1e-6


println("\n bfgs   ")


reset!(nlp)
reinit!(stp)
Lp = 2.0
my_unconstrained_check(nlp, st; kwargs...) = unconstrained_check(nlp, st, pnorm = Lp; kwargs...)
stp.meta.optimality_check = my_unconstrained_check

stats, stp = test_Stp(bfgs_StopLS, nlp, stp=stp, LS_algo=bracket_B)

@info log_row(Any["bfgsSLS-L2", stats.time,  stp.meta.nb_of_stop, stp.current_state.fx, stp.current_state.current_score])
@test norm(stp.current_state.gx, Lp) < 1e-6

reset!(nlp)
reinit!(stp)
Lp = Inf
my_unconstrained_check(nlp, st; kwargs...) = unconstrained_check(nlp, st, pnorm = Lp; kwargs...)
stp.meta.optimality_check = my_unconstrained_check

stats, stp = test_Stp(bfgs_StopLS, nlp, stp=stp, LS_algo=bracket_B)

@info log_row(Any["bfgsSLS-L∞", stats.time,  stp.meta.nb_of_stop, stp.current_state.fx, stp.current_state.current_score])
@test norm(stp.current_state.gx, Lp) < 1e-6

reset!(nlp)
reinit!(stp)

stats, stp = test_Stp(bfgs_Stop, nlp, stp=stp)

@info log_row(Any["bfgsS", stats.time,  stp.meta.nb_of_stop, stp.current_state.fx, stp.current_state.current_score])
@test stp.current_state.current_score < 1e-6





reset!(nlp)


function test_noStp(algo::Function, nlp; kwargs...) where T

    stats = @timed bidon=0
    iter=0
    f=0.0
    normg=0.0
    logger = Logging.ConsoleLogger(stderr,Logging.Warn)
    Logging.with_logger(logger) do 
        stats = @timed iter, f, normg, B = algo(nlp; kwargs...)
    end
        
    return stats, iter, f, normg
end


stats, iter, f, g = test_noStp(bfgs, nlp, scaling = true, maxiter = maxiter, Lp = Inf)



@info log_row(Any["bfgs", stats.time,  iter, f, g])
@test g < 1e-6


println("\n L-bfgs   ")

mem = 70
reset!(nlp)
reinit!(stp)

stats, stp = test_Stp(L_bfgs_StopLS, nlp, stp=stp, LS_algo=bracket_B, mem = mem)

@info log_row(Any["L-bfgsSLS", stats.time,  stp.meta.nb_of_stop, stp.current_state.fx, stp.current_state.current_score])
@test stp.current_state.current_score < 1e-6

reset!(nlp)
reinit!(stp)

stats, stp = test_Stp(L_bfgs_Stop, nlp, stp=stp, mem = mem)

@info log_row(Any["L-bfgsS", stats.time,  stp.meta.nb_of_stop, stp.current_state.fx, stp.current_state.current_score])
@test stp.current_state.current_score < 1e-6


reset!(nlp)



function test_noStp(algo::Function, nlp; kwargs...) where T

    stats = @timed bidon=0
    iter=0
    f=0.0
    normg=0.0
    logger = Logging.ConsoleLogger(stderr,Logging.Warn)
    Logging.with_logger(logger) do 
        stats = @timed iter, f, normg, B = algo(nlp; kwargs...)
    end
        
    return stats, iter, f, normg
end


stats, iter, f, g = test_noStp(L_bfgs, nlp, scaling = true, maxiter = maxiter, Lp = Inf, mem = mem)



@info log_row(Any["L-bfgs", stats.time,  iter, f, g])
@test g < 1e-6


println("\n Newton   ")



reset!(nlp)
reinit!(stp)

stats, stp = test_Stp(Newton_StopLS, nlp, stp=stp)

@info log_row(Any["NwtSLS", stats.time,  stp.meta.nb_of_stop, stp.current_state.fx, stp.current_state.current_score])
@test stp.current_state.current_score < 1e-6


reset!(nlp)
reinit!(stp)

stats, stp = test_Stp(Newton_Stop, nlp, stp=stp)

@info log_row(Any["NwtS", stats.time,  stp.meta.nb_of_stop, stp.current_state.fx, stp.current_state.current_score])
@test stp.current_state.current_score < 1e-6


reset!(nlp)

function test_noStp(algo::Function, nlp; kwargs...) where T

    stats = @timed bidon=0
    iter=0
    f=0.0
    normg=0.0
    logger = Logging.ConsoleLogger(stderr,Logging.Warn)
    Logging.with_logger(logger) do 
        stats = @timed iter, f, normg = algo(nlp; kwargs...)
    end
        
    return stats, iter, f, normg
end


stats, iter, f, g = test_noStp(Newton_Spectral, nlp,  maxiter = maxiter, Lp = Inf, ϵ = 1e-6)



@info log_row(Any["Nwt", stats.time,  iter, f, g])
@test g < 1e-6

