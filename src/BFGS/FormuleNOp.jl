
function  push!(Bop :: BFGSOperator{T, N, F1, F2, F3},
                sk:: Vector{T},
                yk:: Vector{T}) where {T, N, F1, F2, F3}
    
    scaling = Bop.data.scaling
    nsk2 =sk'*sk
    denom = yk'*sk
    B = Bop.data.M
    if (denom > 1.0e-20) # && (mod(iter,n)!=0)
        #self-scaled version aka Luenberger
        By = B*yk
        ytBy = (yk'*By)
        γ = 1.0
        if scaling
            γ = denom / (ytBy)
        end
        
        # math formula, involves matrix multiplications, thus O(n³) complexity
        M = I - sk*yk'/denom
        B1 = γ*M*B*M' + sk*sk'/denom
        B = 0.5*(B1+B1')  # make sure B is symmetric
    else
        @warn "No update, B=I, denom = ", denom
        B = I
    end

    Bop.data.M .= B
    
    return Bop
end

