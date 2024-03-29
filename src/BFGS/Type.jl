export BFGSOperator, InverseBFGSOperator, QNData, BFGSData

using FastClosures

abstract type QNData{T} end

mutable struct BFGSData{T} <: QNData{T}
    M :: Matrix{T}
    scaling::Bool
    Ax :: Vector{T}  # pre allocated for mul5
end

function BFGSData(
    M₀ :: Matrix{T};
    scaling :: Bool = true ) where {T}

    n,  = size(M₀) # add checks for square symmetric matrix

    BFGSData{T}(M₀, scaling, Vector{T}(undef, n))
end


"A type for matrix BFGS approximations."
mutable struct BFGSOperator{T, N <: Integer, F, Ft, Fct} <: AbstractLinearOperator{T}
    nrow::N
    ncol::N
    symmetric::Bool
    hermitian::Bool
    prod!::F    # apply the operator to a vector
    tprod!::Ft    # apply the transpose operator to a vector
    ctprod!::Fct   # apply the transpose conjugate operator to a vector
    #inverse::Bool
    data::QNData{T}
    #data::BFGSData{T}
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

import LinearOperators:has_args5, use_prod5!, isallocated5, storage_type
has_args5(op::BFGSOperator) = true
use_prod5!(op::BFGSOperator) = true
isallocated5(op::BFGSOperator) = true
storage_type(op::BFGSOperator{T}) where {T} = Vector{T}


"""
    InverseBFGSOperator(M₀, n [; scaling=true])
    InverseBFGSOperator(n, [; scaling=true])
Construct a BFGS approximation in inverse form.
"""
function InverseBFGSOperator(M :: Matrix{T}, n :: Int; kwargs...) where {T <: Real}
    kwargs = Dict(kwargs)
    bfgs_data = BFGSData(M; kwargs...)

    function bfgs_multiply(res::AbstractVector,
                           data::BFGSData,
                           x::AbstractArray,
                           αm,
                           βm::T2,
                           ) where T2

        q = data.Ax  # pre allocated
        q .= data.M * x

        # mul5 stuff
        if βm == zero(T2)
            res .= αm .* q
        else
            res .= αm .* q .+ βm .* res
        end

        return res
    end

    prod! = @closure (res, x, α, β) -> bfgs_multiply(res, bfgs_data, x, α, β)
    return BFGSOperator{T}(n, n, true, true, prod!, prod!, prod!, bfgs_data)
end

function InverseBFGSOperator(T, n :: N; kwargs...) where {N <: Integer}
    Eye = Matrix{T}(I,n,n)
    InverseBFGSOperator(Eye, n; kwargs...)
end
