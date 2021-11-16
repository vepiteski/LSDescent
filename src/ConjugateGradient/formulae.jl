export formula_FR, formula_PR, formula_HS, formula_HZ

function formula_FR(∇f,∇ft,s,d)

    β = (∇ft⋅∇ft)/(∇f⋅∇f)
    
    return β
end

function formula_PR(∇f,∇ft,s,d)

    y = ∇ft - ∇f
    β = (∇ft⋅y)/(∇f⋅∇f)
    
    return β
end

function formula_HS(∇f,∇ft,s,d)

    y = ∇ft - ∇f
    β = (∇ft⋅y)/(d⋅y)
    
    return β
end

function formula_HZ(∇f,∇ft,s,d)

    y = ∇ft - ∇f
    n2y = y⋅y
    β1 = (y⋅d)
    β = ((y-2*d*n2y/β1)⋅∇ft)/β1
    
    return β
end
