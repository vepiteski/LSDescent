""" Conjugate gradient algorithm using a line search supporting stopping
    Generic version receiving a conjugacy formula
    kwargs may specify options for the line search.
"""  
function CG_generic(nlp       :: AbstractNLPModel;
                    x         :: Vector{T}=copy(nlp.meta.x0),
                    stp       :: NLPStopping = NLPStopping(nlp,
                                                           NLPAtX(nlp.meta.x0)),
                    scaling   :: Bool = true,
                    LS_algo   :: Function = bracket{T},
                    LS_logger :: AbstractLogger = Logging.NullLogger(),
                    CG_formula :: Function = formula_HZ,
                    kwargs...      # eventually options for the line search
                    ) where T
    
    
    @info log_header([:iter, :f, :dual, :step, :slope], [Int, T, T, T, T],
                     hdr_override=Dict(:f=>"f(x)", :dual=>"‖∇f‖", :slope=>"IterLS"))
    
    n = nlp.meta.nvar
    
    f, ∇f = objgrad(nlp, x)
    
    xt = similar(x)
    ∇ft = similar(∇f)
    
    τ₀ = 0.05
    τ₁ = 0.16
    
    ϕ, ϕstp = prepare_LS(stp, x, ∇f, τ₀, f, ∇f)
    
    OK = update_and_start!(stp, x = x, fx = f, gx = ∇f)

    @info log_row(Any[0, f, norm(∇f)])
    
    β = 0.0
    d = zeros(n)
    scale = 1.0
    
    while !OK
        d = - ∇f + β*d
        
        slope = ∇f⋅d
        if slope > 0.0  # restart with negative gradient
            d = - ∇f
          slope =  ∇f⋅d
        end

        # Simple line search call
        # returns  t, xt = x + t*d,  ft=f(xt), ∇ft = ∇f(xt)

        t, xt, ft, ∇ft = linesearch(ϕ, ϕstp, x, d*scale, f, ∇f, τ₀, τ₁, logger = LS_logger, algo = LS_algo; kwargs...)
        fail_sub_pb = ~ϕstp.meta.optimal

        if fail_sub_pb
            OK = true
            stp.meta.fail_sub_pb = true
        else
            #move on
            s = xt - x
            y = ∇ft - ∇f
            β = 0.0
            #if (∇ft⋅∇f) < 0.75 * (∇ft⋅∇ft)   # Powell restart
                β = CG_formula(∇f,∇ft,s,d)
            #end
            if scaling
                scale = (y⋅s) / (y⋅y)
            end
            if scale <= 0.0
                scale = 1.0
            end
            x .= xt
            f = ft
            ∇f .= ∇ft
            OK = update_and_stop!(stp, x = x, gx = ∇f, fx = f)
        end
        
        @info log_row(Any[stp.meta.nb_of_stop, f, norm(∇f), t, ϕ.counters.neval_obj])
    end
    
    if !stp.meta.optimal
        @warn status(stp,list=true)
        @debug x
    end
    
    return stp
end
