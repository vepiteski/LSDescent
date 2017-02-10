export NwtdirectionSpectral

function NwtdirectionSpectral(H,g;verbose::Bool=false)
    Δ = ones(g)
    V = ones(H)
    try
        Δ, V = eig(H)
    catch
        Δ, V = eig(H + eye(H))
    end
    ϵ2 =  1.0e-8 
    Γ = 1.0 ./ max(abs(Δ),ϵ2)
    
    d = - (V * diagm(Γ) * V') * (g)
    return d
end
