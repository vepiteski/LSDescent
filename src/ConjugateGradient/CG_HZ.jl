export CG_HZ

function CG_HZ(nlp :: AbstractNLPModel;
               x   :: Vector{T}=copy(nlp.meta.x0),
               stp :: NLPStopping = NLPStopping(nlp,
                                                NLPAtX(nlp.meta.x0)),
               scaling   :: Bool = true,
               LS_algo   :: Function = bracket,
               LS_logger :: AbstractLogger = Logging.NullLogger(),
               kwargs...) where T

    return CG_generic(nlp;
                      x = x,
                      stp=stp,
                      scaling = scaling,
                      LS_algo = LS_algo,
                      LS_logger = LS_logger,
                      CG_formula  = formula_HZ,
                      kwargs...)
end
