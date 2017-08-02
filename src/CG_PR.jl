export CG_PR

function CG_PR(nlp :: AbstractNLPModel;
               stp :: TStopping = TStopping(),
               verbose :: Bool=false,
               verboseLS :: Bool = false,
               mem :: Int=5,
               linesearch :: Function = Newarmijo_wolfe,
               scaling :: Bool = true,
               kwargs...)

    return CG_generic(nlp;
                      stp=stp,
                      verbose = verbose,
                      verboseLS = verboseLS,
                      mem = mem,
                      linesearch  = linesearch,
                      CG_formula  = formula_PR,
                      scaling  = scaling,
                      kwargs...)
end
