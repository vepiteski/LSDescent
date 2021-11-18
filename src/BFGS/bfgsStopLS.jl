""" BFGS algorithm using a line search supporting stopping
    The initial B₀ may be a matrix, an identity or a operator.
    Using the operator implements L-BFGS
    kwargs may specify options for the line search.
"""  
function bfgs_StopLS(nlp       :: AbstractNLPModel;
                     x         :: Vector{T}=copy(nlp.meta.x0),
                     stp       :: NLPStopping = NLPStopping(nlp,
                                                      NLPAtX(nlp.meta.x0)),
                     scaling   :: Bool = true,
                     Lp        :: Real = 2.0, # norm Lp 
                     LS_algo   :: Function = bracket,
                     LS_logger :: AbstractLogger = Logging.NullLogger(),
                     B₀        :: Union{AbstractLinearOperator,
                                        AbstractMatrix,
                                        UniformScaling{T},
                                        Nothing}              = nothing,
                     kwargs...      # eventually options for the line search
                     ) where T
    
    my_unconstrained_check(nlp, st; kwargs...) = unconstrained_check(nlp, st, pnorm = Lp; kwargs...)
    stp.meta.optimality_check = my_unconstrained_check
    
    @info log_header([:iter, :f, :dual, :step, :slope], [Int, T, T, T, T],
                     hdr_override=Dict(:f=>"f(x)", :dual=>"‖∇f‖", :slope=>"∇fᵀd"))
    
    f = obj(nlp,x)
    ∇f = grad(nlp, x)

    xt = similar(x)
    ∇ft = similar(∇f)

    B = one(eltype(x))*I
    if B₀ != nothing
        B = B₀
    end

    τ₀ = 0.0005
    τ₁ = 0.9999
    
    ϕ, ϕstp = prepare_LS(stp, x, ∇f, τ₀, f, ∇f)

    OK = update_and_start!(stp, x = x, fx = f, gx = ∇f)
    #update_and_stop!(stp,  x = x, fx = f, gx = ∇f)
    @info log_row(Any[0, f, norm(∇f)])

    while !OK
        d = - B*∇f

        # Simple line search call
        # returns  t, xt = x + t*d,  ft=f(xt), ∇ft = ∇f(xt)

        t, xt, ft, ∇ft = linesearch(ϕ, ϕstp, x, d, f, ∇f, τ₀, τ₁, logger = LS_logger, algo = LS_algo; kwargs...)
        fail_sub_pb = ~ϕstp.meta.optimal
                
        if fail_sub_pb
            OK = true
            stp.meta.fail_sub_pb = true
        else
            # Update BFGS approximation.
            B = push!(B, t * d, ∇ft - ∇f)
            #B = push!(B, t * d, ∇ft - ∇f, scaling)
            
            #move on
            x .= xt
            f = ft
            ∇f .= ∇ft
            
            OK = update_and_stop!(stp, x = x, gx = ∇f, fx = f)
        end
        norm∇f = stp.current_state.current_score
        @info log_row(Any[stp.meta.nb_of_stop, f, norm∇f, t, ϕ.counters.neval_obj])
    end
    
    if !stp.meta.optimal
        @warn status(stp,list=true)
        @debug x
    end
    
    return tuple(stp, B)
end


""" L-BFGS wrapper of the gereral BFGS implementation.
"""
function L_bfgs_StopLS(nlp :: AbstractNLPModel;
                       x :: Vector{T}=copy(nlp.meta.x0),
                       stp :: NLPStopping = NLPStopping(nlp,
                                                        NLPAtX(nlp.meta.x0)),
                       mem :: Int = 5,
                       scaling :: Bool = true,
                       LS_algo   :: Function = bracket,
                       LS_logger :: AbstractLogger = Logging.NullLogger(),
                       kwargs...      # eventually options for the line search
                       ) where T

    @debug "U_Solver = L_bfgs_StopLS"
    n = nlp.meta.nvar
    B₀ =  InverseLBFGSOperator(Float64, n, mem=mem, scaling=scaling)
    
    return bfgs_StopLS(nlp; x=x, stp=stp, scaling=scaling, LS_logger=LS_logger, LS_algo=LS_algo, B₀=B₀, kwargs...)
end


 
