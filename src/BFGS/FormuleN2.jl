import Base.push!

function push!(B :: Union{UniformScaling{T},Matrix{T}},
               sk:: Vector{T},
               yk:: Vector{T}) where T,
#               scaling :: Bool = true) where T

    scaling = true
    denom = yk'*sk
    if (denom > 1.0e-20) 
        #self-scaled version aka Luenberger

        #divide vector yk by denom 
        By = B*(yk/denom)

        ytBy = (yk'*By)
        γ = 1.0
        if scaling
            γ = T(1) / (ytBy)
        end

        # refactor to yield O(n²) complexity... Much more efficient!
        Byst = By*sk'

        #divide vector sk by denom
        sst = (sk/denom)*sk'

        #B = γ * (B - (Byst' + Byst)/denom + (ytBy/denom)*sst/denom)  + sst/denom
        # avoids dividing matrices by denom
        #
        B = γ * (B  - (Byst' + Byst) + ytBy*sst)  + sst
        B = T(0.5)*(B + B')  # make sure B is symmetric

        # Equivalent math formulation, for testing purposes.
        # math formula, involves matrix multiplications, thus O(n³) complexity
        #M = I - sk*yk'/denom
        #B1 = γ * M*B1*M' + sk*sk'/denom
        #B1 = T(0.5)*(B1 + B1')  # make sure B is symmetric
        #println( " In push! différence entre B et B1 : " , norm(B - B1))
    else
        @warn "No update,  denom = ", denom
        #B = Matrix(eltype(yk).(I(length(yk))))
    end
    
    return B
end

