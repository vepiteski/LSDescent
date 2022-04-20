export NwtdirectionSpectral, NewtonSpectralAbs

function NwtdirectionSpectral(H, ∇f; scale_abs :: Bool = true,
                              γ = 1e-6)
    Δ, O = eigen(H)
    # Boost negative values of Δ to 1e-8
    # devise an adaptative value for γ
    #γ = 1e-6
    n = length(Δ)
    if scale_abs   
        #D = abs.(Δ) + max.((γ .- abs.(Δ)), 0.0) .*ones(n)
        D = max.(abs.(Δ), γ)
    else
        D = max.(Δ, γ)
    end
    d = - O*diagm(1.0 ./ D)*O'*∇f
    #Δ = ones(g)
    #V = ones(H)
    #try
    #    Δ, V = eig(H)
    #catch
    #    Δ, V = eig(H + eye(H))
    #end
    #ϵ2 =  1.0e-8 
    #Γ = 1.0 ./ max.(abs.(Δ),ϵ2)
    
    #d = - (V * diagm(Γ) * V') * (g)
    return d
end

function NewtonSpectralAbs(nlp :: AbstractNLPModel;
                           x :: Vector{T}=copy(nlp.meta.x0),
                           stp :: NLPStopping = NLPStopping(nlp,
                                                            NLPAtX(nlp.meta.x0)),
                           kwargs...) where T
    return  Newton_G_Stop(nlp;
                   x = x,
                   stp = stp,
                   NwtDirection = NwtdirectionSpectral,
                   hessian_rep = hessian_dense,
                   kwargs...)
end


function NewtonSpectralAbsLS(nlp :: AbstractNLPModel;
                               x :: Vector{T}=copy(nlp.meta.x0),
                               stp :: NLPStopping = NLPStopping(nlp,
                                                                NLPAtX(nlp.meta.x0)),
                               LS_algo   :: Function = bracket,
                               LS_logger :: AbstractLogger = Logging.NullLogger(),
                               kwargs...) where T
      return  Newton_G_StopLS(nlp;
                              x = x,
                              stp = stp,
                              LS_algo = LS_algo,
                              LS_logger = LS_logger,
                              NwtDirection = NwtdirectionSpectral,
                              hessian_rep = hessian_dense,
                              kwargs...)
  end


