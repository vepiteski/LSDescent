mutable struct ChBFGSData{T} <: QNData{T}
    C :: Cholesky{T}
    scaling::Bool
    Ax :: Vector{T}  # pre allocated for mul5
end

function ChBFGSData(
    M₀ :: Matrix{T};
    scaling :: Bool = true ) where {T}

    n,  = size(M₀) # add checks for square symmetric matrix ≻ 0
    C₀ = cholesky(M₀)
    
    ChBFGSData{T}(C₀, scaling, Vector{T}(undef, n))
end


BFGSOperator{T}(
    nrow :: N,
    ncol :: N,
    symmetric :: Bool,
    hermitian :: Bool,
    prod!::F,
    tprod!::Ft,
    ctprod!::Fct,
    data::ChBFGSData{T},
) where {T, N, F, Ft, Fct} = BFGSOperator{T, N, F, Ft, Fct}(
    nrow,
    ncol,
    symmetric,
    hermitian,
    prod!,
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
function ChBFGSOperator(M :: Matrix{T}; kwargs...) where {T <: Real}
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

        return res
    end
    
    prod! = @closure (res, x, α, β) -> Chbfgs_multiply(res, Ch_bfgs_data, x, α, β)
    return BFGSOperator{T}(n, n, true, true, prod!, prod!, prod!, Ch_bfgs_data)
end

function ChBFGSOperator(T, n :: N; kwargs...) where {N <: Integer}
    Eye = Matrix{T}(I,n,n)
    ChBFGSOperator(Eye; kwargs...)
end



#function Matrix(Op::InverseBFGSOperator)
#    return Op.data.M
#end
