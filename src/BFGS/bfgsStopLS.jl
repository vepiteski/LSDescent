export bfgs_StopLS, L_bfgs_StopLS, M_bfgs_StopLS, Ch_bfgs_StopLS

include("AcceptAll.jl")

""" BFGS algorithm using a line search supporting stopping
    The initial B₀ may be a matrix, an identity or a operator.
    Using the operator implements L-BFGS
    kwargs may specify options for the line search.
"""  
function bfgs_StopLS(nlp       :: AbstractNLPModel{T, S};
                     x         :: S = copy(nlp.meta.x0),
                     stp       :: NLPStopping = NLPStopping(nlp,
                                                      NLPAtX(nlp.meta.x0)),
                     scaling   :: Bool = true,
                     LS_algo   :: Function = bracket_s2N,
                     LS_logger :: AbstractLogger = Logging.NullLogger(),
                     B₀        :: Union{AbstractLinearOperator,
                                        AbstractMatrix,
                                        UniformScaling{T},
                                        Nothing}              = nothing,
                     kwargs...      # eventually options for the line search
                     ) where {T, S}
    
    
    @info log_header([:iter, :f, :dual, :step, :nBtrk], [Int, T, T, T, T],
                     hdr_override=Dict(:f=>"f(x)", :dual=>"‖∇f‖"))

    n = length(x)
    f = obj(nlp,x)
    ∇f = grad(nlp, x)

    xt = similar(x)
    ∇ft = similar(∇f)

    B = AcceptAll(T, B₀)

    #B = InverseBFGSOperator(T, n)
    #B = one(eltype(x))*I

    #if B₀ != nothing
    #    if isa(B₀, AbstractMatrix)
    #        B = InverseBFGSOperator(B)
    #    elseif isa(B₀, LinearOperator)
    #        B = B₀
    #    end
    #end

    # TO DO: systematically convert to an operator (encapsulate matrices)
    
    stp.stopping_user_struct["BFGS"] = B

    τ₀ = 0.0005
    τ₁ = 0.9999
    
#    ϕ, ϕstp = prepare_LS(stp, x, ∇f, τ₀, f, ∇f)

    OK = update_and_start!(stp, x = x, fx = f, gx = ∇f)
    #update_and_stop!(stp,  x = x, fx = f, gx = ∇f)
    ϕ, ϕstp = prepare_LS(stp, x, ∇f, τ₀, f, ∇f)
    @info log_row(Any[0, f, norm(∇f)])

    while !OK
        d = - B*∇f

        # Simple line search call
        # returns  t, xt = x + t*d,  ft=f(xt), ∇ft = ∇f(xt)
        reset!(ϕ)
        t, xt, ft, ∇ft = linesearch(ϕ, ϕstp, x, d, f, ∇f, τ₀, τ₁, logger = LS_logger, algo = LS_algo; kwargs...)
        fail_sub_pb = ~ϕstp.meta.optimal
                
        if fail_sub_pb
            OK = true
            stp.meta.fail_sub_pb = true
        #end
        # try anyway the step
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
        #@show f
        @info log_row(Any[stp.meta.nb_of_stop, f, norm(∇f), t, ϕ.counters.neval_obj])
    end
    
    if !stp.meta.optimal
        @warn status(stp,list=true)
        @debug x
    end
    
    #return tuple(stp, B)
    return stp
end


""" L-BFGS wrapper of the gereral BFGS implementation.
"""
function L_bfgs_StopLS(nlp :: AbstractNLPModel{T, S};
                       x :: S = copy(nlp.meta.x0),
                       stp :: NLPStopping = NLPStopping(nlp,
                                                        NLPAtX(nlp.meta.x0)),
                       mem :: Int = 5,
                       scaling :: Bool = true,
                       LS_algo   :: Function = bracket_B,
                       LS_logger :: AbstractLogger = Logging.NullLogger(),
                       kwargs...      # eventually options for the line search
                       ) where {T, S}

    @debug "U_Solver = L_bfgs_StopLS"
    n = nlp.meta.nvar
    B₀ =  InverseLBFGSOperator(Float64, n, mem=mem, scaling=scaling)
    
    return bfgs_StopLS(nlp; x=x, stp=stp, scaling=scaling, LS_logger=LS_logger, LS_algo=LS_algo, B₀=B₀, kwargs...)
end

""" M-BFGS wrapper of the gereral BFGS implementation. M == Matrix
"""
function M_bfgs_StopLS(nlp :: AbstractNLPModel;
                       x :: Vector{T}=copy(nlp.meta.x0),
                       stp :: NLPStopping = NLPStopping(nlp,
                                                        NLPAtX(nlp.meta.x0)),
                       scaling :: Bool = true,
                       LS_algo   :: Function = bracket_B,
                       LS_logger :: AbstractLogger = Logging.NullLogger(),
                       kwargs...      # eventually options for the line search
                       ) where T

    @debug "U_Solver = M_bfgs_StopLS"
    n = nlp.meta.nvar
    B₀ =  InverseBFGSOperator(Float64, n, scaling=scaling)
    
    return bfgs_StopLS(nlp; x=x, stp=stp, scaling=scaling, LS_logger=LS_logger, LS_algo=LS_algo, B₀=B₀, kwargs...)
end

""" Ch-BFGS wrapper of the gereral BFGS implementation. Ch == Cholesky
"""
function Ch_bfgs_StopLS(nlp :: AbstractNLPModel;
                       x :: Vector{T}=copy(nlp.meta.x0),
                       stp :: NLPStopping = NLPStopping(nlp,
                                                        NLPAtX(nlp.meta.x0)),
                       scaling :: Bool = true,
                       LS_algo   :: Function = bracket_B,
                       LS_logger :: AbstractLogger = Logging.NullLogger(),
                       kwargs...      # eventually options for the line search
                       ) where T

    @debug "U_Solver = M_bfgs_StopLS"
    n = nlp.meta.nvar
    include("FormuleChOp.jl")
    include("TypeChol.jl")

    B₀ = ChBFGSOperator(Float64, n; scaling = scaling);
    #B₀ =  InverseBFGSOperator(Float64, n, scaling=scaling)
    
    return bfgs_StopLS(nlp; x=x, stp=stp, scaling=scaling, LS_logger=LS_logger, LS_algo=LS_algo, B₀=B₀, kwargs...)
end


 
