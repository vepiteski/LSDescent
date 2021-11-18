export CG_HS

function CG_HS(nlp :: AbstractNLPModel;
               x   :: Vector{T}=copy(nlp.meta.x0),
               stp :: NLPStopping = NLPStopping(nlp,
                                                NLPAtX(nlp.meta.x0)),
               scaling   :: Bool = true,
               LS_algo   :: Function = bracket{T},
               LS_logger :: AbstractLogger = Logging.NullLogger(),
               kwargs...)

    return CG_generic(nlp;
                      x = x,
                      stp=stp,
                      scaling = scaling,
                      LS_algo = LS_algo,
                      LS_logger = LS_logger,
                      CG_formula  = formula_HS,
                      kwargs...)
end
