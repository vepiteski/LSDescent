using FastClosures

export BFGSdata, BFGSOperator, InverseBFGSOperator

mutable struct BFGSData{T}
    M :: Matrix{T}
    scaling::Bool
end

function BFGSData(
    M₀ :: Matrix{T};
    scaling :: Bool = true ) where {T}

    BFGSData{T}(M₀, scaling)
end


"A type for limited-memory BFGS approximations."
mutable struct BFGSOperator{T, N <: Integer, F, Ft, Fct} <: AbstractLinearOperator{T}
    nrow::N
    ncol::N
    symmetric::Bool
    hermitian::Bool
    prod!::F    # apply the operator to a vector
    tprod!::Ft    # apply the transpose operator to a vector
    ctprod!::Fct   # apply the transpose conjugate operator to a vector
    #inverse::Bool
    data::BFGSData{T}
    nprod::N
    ntprod::N
    nctprod::N
end

BFGSOperator{T}(
    nrow :: N,
    ncol :: N,
    symmetric :: Bool,
    hermitian :: Bool,
    prod!::F,
    tprod!::Ft,
    ctprod!::Fct,
    data::BFGSData{T},
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
function InverseBFGSOperator(M :: Matrix{T}; kwargs...) where {T <: Real}
    kwargs = Dict(kwargs)
    bfgs_data = BFGSData(M; kwargs...)

    function bfgs_multiply(res::AbstractVector,
                           data::BFGSData,
                           x::AbstractArray,
                           αm,
                           βm::T2,
                           ) where T2
        
        res .= data.M * x
        #@show data.M
        #@show x
        #@show res
        return res
    end
    
    prod! = @closure (res, x, α, β) -> bfgs_multiply(res, bfgs_data, x, α, β)
    return BFGSOperator{T}(n, n, true, true, prod!, prod!, prod!, bfgs_data)
end

function InverseBFGSOperator(T, n :: N; kwargs...) where {N <: Integer}
    Eye = Matrix{T}(I,n,n)
    InverseBFGSOperator(Eye; kwargs...)
end



#function Matrix(Op::InverseBFGSOperator)
#    return Op.data.M
#end
