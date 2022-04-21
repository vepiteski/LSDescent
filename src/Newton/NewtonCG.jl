export NwtdirectionCG, NewtonCG, NewtonCGLS
using Krylov

function NwtdirectionCG(H, ∇f; τ = 0.5, kwargs...)
    e=1e-1
    n = length(∇f)
    #τ = 0.5 # need parametrization
    cgtol = max(e, min(0.7, 0.01 * norm(∇f)^(1.0 + τ)))
    
    (d, cg_stats) = cg(H, -∇f,
                       atol=cgtol, rtol=0.0,
                       itmax=max(2 * n, 50),
                       linesearch = true,
                       kwargs...)
    
    return d
end

function NewtonCG(nlp :: AbstractNLPModel;
                  x :: Vector{T}=copy(nlp.meta.x0),
                  stp :: NLPStopping = NLPStopping(nlp,
                                                   NLPAtX(nlp.meta.x0)),
                  kwargs...) where T
    return  Newton_G_Stop(nlp;
                          x = x,
                          stp = stp,
                          NwtDirection = NwtdirectionCG,
                          hessian_rep = hessian_operator,
                          kwargs...)
end


function NewtonCGLS(nlp :: AbstractNLPModel;
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
                          NwtDirection = NwtdirectionCG,
                          hessian_rep = hessian_operator,
                          kwargs...)
end
