export NwtdirectionSpectral, NewtonSpectralAbs

function NwtdirectionSpectral(H,g;verbose::Bool=false)
    Δ = ones(g)
    V = ones(H)
    try
        Δ, V = eig(H)
    catch
        Δ, V = eig(H + eye(H))
    end
    ϵ2 =  1.0e-8 
    Γ = 1.0 ./ max.(abs.(Δ),ϵ2)
    
    d = - (V * diagm(Γ) * V') * (g)
    return d
end

function NewtonSpectralAbs(nlp :: AbstractNLPModel;
                           stp :: TStopping=TStopping(),
                           verbose :: Bool=false,
                           verboseLS :: Bool = false,
                           linesearch :: Function = Newarmijo_wolfe,
                           kwargs...)
    return  Newton(nlp;
                   stp = stp,
                   verbose = verbose,
                   verboseLS = verboseLS,
                   linesearch = linesearch,
                   Nwtdirection = NwtdirectionSpectral,
                   hessian_rep = hessian_dense,
                   kwargs...)
end

    
