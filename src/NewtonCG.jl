export NwtdirectionCG, NewtonCG

function NwtdirectionCG(H,∇f;verbose::Bool=false)
    e=1e-6
    n = length(∇f)
    τ = 0.5 # need parametrization
    cgtol = max(e, min(0.7, 0.01 * norm(∇f)^(1.0 + τ)))
    
    (d, cg_stats) = cgTN(H, -∇f,
                       atol=cgtol, rtol=0.0,
                       itmax=max(2 * n, 50),
                       verbose=verbose)
    
    return d
end

NewtonCG = Newton
