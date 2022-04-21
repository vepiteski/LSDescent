""" Spectral Newton algorithm using a line search supporting stopping
"""
function Newton_G_StopLS(nlp :: AbstractNLPModel;
                         x :: Vector{T}=copy(nlp.meta.x0),
                         stp :: NLPStopping = NLPStopping(nlp,
                                                          NLPAtX(nlp.meta.x0)),
                         LS_algo   :: Function = bracket,
                         LS_logger :: AbstractLogger = Logging.NullLogger(),                    
                         NwtDirection :: Function = NwtdirectionSpectral,
                         hessian_rep :: Function = hessian_dense,
                         kwargs...      # eventually options for the line search
                         ) where T
   
    
    n = nlp.meta.nvar

     @info log_header([:iter, :f, :dual, :step, :slope], [Int, T, T, T, T],
                     hdr_override=Dict(:f=>"f(x)", :dual=>"‖∇f‖", :slope=>"∇fᵀd"))
    f ::T  = obj(nlp,x)
    ∇f :: Vector{T} = grad(nlp, x)

    OK = update_and_start!(stp, x = x, fx = f, gx = ∇f)

    τ₀ = 0.0005
    τ₁ = 0.9999
    d = similar(x)
    ϕ, ϕstp = prepare_LS(stp, x, d, τ₀, f, ∇f)

    @info log_row(Any[0, f, norm(∇f)])

    while !OK

        H = hessian_rep(nlp, x)
        
        d = NwtDirection(H, ∇f; kwargs...)

        hp0 = ∇f⋅d
        # Simple line search call        

        t, x, f, ∇f = linesearch(ϕ, ϕstp, x, d, f, ∇f, τ₀, τ₁, logger = LS_logger, algo = LS_algo)#; kwargs...)
        fail_sub_pb = ~ϕstp.meta.optimal
        
        if fail_sub_pb
            OK = true
            stp.meta.fail_sub_pb = true
        else
            OK = update_and_stop!(stp, x = x, gx = ∇f, fx=f)
        end
        @info log_row(Any[stp.meta.nb_of_stop, f, norm(∇f), t, hp0])
    end
    
    if !stp.meta.optimal
        #@warn "Optimalité non atteinte"
        @warn status(stp, list=true)
    end
    
    return stp
end
