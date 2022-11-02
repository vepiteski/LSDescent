export NwtdirectionSpectral, NewtonSpectralAbs, NewtonSpectralAbsLS

function NwtdirectionSpectral(H, ∇f; scale_abs :: Bool = true,
                              γ = 1e-6)
    Δ, O = eigen(Hermitian(H))
    # devise an adaptative value for γ ? akin to Levenberg-Marquardt?
    n = length(Δ)
    if scale_abs   
        D = max.(abs.(Δ), γ)
    else
        D = max.(Δ, γ)
    end
    d = - O*diagm(1.0 ./ D)*O'*∇f
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


