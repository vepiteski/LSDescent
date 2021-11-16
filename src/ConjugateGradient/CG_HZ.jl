export CG_HZ

function CG_HZ(nlp :: AbstractNLPModel;
               stp :: TStopping = TStopping(),
               verbose :: Bool=false,
               verboseLS :: Bool = false,
               linesearch :: Function = Newarmijo_wolfe,
               scaling :: Bool = true,
               kwargs...)

    return CG_generic(nlp;
                      stp=stp,
                      verbose = verbose,
                      verboseLS = verboseLS,
                      linesearch  = linesearch,
                      CG_formula  = formula_HZ,
                      scaling  = scaling,
                      kwargs...)
end
