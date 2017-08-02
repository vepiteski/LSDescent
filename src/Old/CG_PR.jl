export CG_PR

function CG_PR(nlp :: AbstractNLPModel;
               atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
               verbose :: Bool=false,
               verboseLS :: Bool = false,
               mem :: Int=5,
               linesearch :: Function = Newarmijo_wolfe,
               scaling :: Bool = true,
               kwargs...)

    return CG_generic(nlp;
                      atol=atol, rtol =rtol,
                      verbose = verbose,
                      verboseLS = verboseLS,
                      mem = mem,
                      linesearch  = linesearch,
                      CG_formula  = formula_PR,
                      scaling  = scaling,
                      kwargs...)
end
