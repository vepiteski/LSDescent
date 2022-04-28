import Base.push!

function  push!(Bop :: ChBFGSOperator{T, N, F1, F2, F3},
                sk:: Vector{T},
                yk:: Vector{T}) where {T, N, F1, F2, F3}
    @info "Dans formuleNop Cholesky"
    scaling = Bop.data.scaling
    denom = yk'*sk
    
    #C = Bop.data.C    prod!,
    tprod!,
    ctprod!,
    data,
    0,
    0,
    0,
)

import LinearOperators:has_args5, use_prod5!, isallocated5
has_args5(op::BFGSOperator) = true
use_prod5!(op::BFGSOperator) = true
isallocated5(op::BFGSOperator) = true



"""
    InverseBFGSOperator(M₀, n [; scaling=true])
    InverseBFGSOperator(n, [; scaling=true])
Construct a BFGS approximation in inverse form. 
"""
function ChBFGSOperator(M :: Matrix{T}, n :: Int; kwargs...) where {T <: Real}
    kwargs = Dict(kwargs)
    Ch_bfgs_data = ChBFGSData(M; kwargs...)

    function Chbfgs_multiply(res::AbstractVector,
                           data::ChBFGSData,
                           x::AbstractArray,
                           αm,
                           βm::T2,
                           ) where T2

        q = data.Ax  # pre allocated
        # name confusion, the Cholesky factors are used to represent
        # the INVERSE operator, thus multiplication is indeed C \ x
        q .= data.C \ x
        # mul5 stuff
        if βm == zero(T2)
            res .= αm .* q
        else
            res .= αm .* q .+ βm .* res
        end
    # try to work inplace
    (;C) = Bop.data

    if (denom > eps(T)) # && (mod(iter,n)!=0)
        #self-scaled version aka Luenberger
        # setup vectors v₁ and v₂
        v₁ = yk / sqrt(denom)
        # à suivre, on n'a pas directement Hk
        # Gill&Murray supposent qu'on a g et α (param du line search)
        # et donc dans ce cas, Hk*sk = α*g évite une multiplication
        # matricielle

        # scaling
        γ = 1.0
        Hs = C.L*C.L'*sk
        stHs = sk'*Hs
        By = C\yk
        ytBy = yk'*By
        if scaling
            # γ = denom / (ytBy)
             γ =  (ytBy) / denom  #OK

            # γ = denom / dot(yk,yk) #NON
            # γ = dot(sk,sk) / denom #NON
            # γ =  (stHs) / denom
            # γ =  denom / (stHs) #OK
            # γ = denom / dot(sk,sk) # non
            # γ =  dot(yk,yk) / denom
            #@show γ
        end
        sqrtγ = sqrt(γ) # will scale factors, so squared root
        # scale Hk in the update formula
        C.factors[:,:] .= (sqrtγ).*C.factors   # scaling of Hk itself
        v₂ = (Hs / sqrt(stHs))*(sqrtγ) # scaling γ for v₂'v₂
        # update the Cholesky factorization
        lowrankupdate!(C, v₁)
        lowrankdowndate!(C, v₂)

    else
        @warn "No update, denom = ", denom
        #B = I
    end
    
    #Should not be needed
    #Bop.data.C = copy(C)

    return Bop
end

