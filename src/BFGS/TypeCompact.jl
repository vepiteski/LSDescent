export CompactInverseBFGSOperator

import Base.push!
import LinearOperators: has_args5, use_prod5!, isallocated5

using FastClosures
using LinearAlgebra

"A data type to hold information relative to compact LBFGS operator."
mutable struct CompactInverseBFGSData{T,I<:Integer}
    k::I
    mem::I
    scaling::Bool
    scaling_factor::T
    S::AbstractArray{T}
    Y::AbstractArray{T}
    R::AbstractArray{T}
    Q::AbstractArray{T}
    SR::AbstractArray{T}
    indices::AbstractVector{I}
end

function CompactInverseBFGSData(
    T::DataType,
    n::I;
    mem::I=5,
    scaling::Bool=false
) where {I<:Integer}
    CompactInverseBFGSData{T,I}(
        0,                                    # iteration number
        max(mem, 1),                          # memory
        scaling,                              # scaling
        convert(T, 1),                        # scaling factor
        zeros(n, mem),                        # S matrix
        zeros(n, mem),                        # Y matrix
        zeros(mem, mem),                      # R matrix
        zeros(mem, mem),                      # D + scaling * YtY matrix
        zeros(n,0),                           # SR
        zeros(Int, 0)                         # indices
    )
end

CompactInverseBFGSData(n::I; kwargs...) where {I<:Integer} = CompactInverseBFGSData(Float64, n; kwargs...)

"A type for compact representation of limited-memory BFGS approximations."
mutable struct CompactInverseBFGSOperator{T,I<:Integer,F,Ft,Fct} <: AbstractLinearOperator{T}
    nrow::I
    ncol::I
    symmetric::Bool
    hermitian::Bool
    prod!::F    # apply the operator to a vector
    tprod!::Ft    # apply the transpose operator to a vector
    ctprod!::Fct   # apply the transpose conjugate operator to a vector
    data::CompactInverseBFGSData{T,I}
    nprod::I
    ntprod::I
    nctprod::I
end

CompactInverseBFGSOperator{T}(
    nrow::I,
    ncol::I,
    symmetric::Bool,
    hermitian::Bool,
    prod!::F,
    tprod!::Ft,
    ctprod!::Fct,
    data::CompactInverseBFGSData{T,I},
) where {T,I<:Integer,F,Ft,Fct} = CompactInverseBFGSOperator{T,I,F,Ft,Fct}(
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

has_args5(op::CompactInverseBFGSOperator) = true
use_prod5!(op::CompactInverseBFGSOperator) = true
isallocated5(op::CompactInverseBFGSOperator) = true

"""
        CompactInverseBFGSOperator(T, n, [mem=5])
        CompactInverseBFGSOperator(n, [mem=5])
Construct a compact representation of limited-memory BFGS approximation.
If the type `T` is omitted, then `Float64` is used.
"""
function CompactInverseBFGSOperator(T::DataType, n::I; kwargs...) where {I<:Integer}
    compact_lbfgs_data = CompactInverseBFGSData(T, n; kwargs...)

    function compact_lbfgs_multiply(
        res::AbstractVector,
        data::CompactInverseBFGSData,
        x::AbstractArray,
        αm,
        βm::T2,
    ) where {T2}

        q = similar(res)
        # Read Operator struct
        # (; scaling_factor, indices, SR) = data
        scaling_factor, indices, SR = data.scaling_factor, data.indices, data.SR

        # Multiply operator with a vector.
        compute_inner_prod!(q, scaling_factor, SR,
                            view(data.Y, :, indices),
                            view(data.Q, indices, indices),
                            x)

        if βm == zero(T2)
            res .= αm .* q
        else
            res .= αm .* q .+ βm .* res
        end
    end

    prod! = @closure (res, x, α, β) -> compact_lbfgs_multiply(res, compact_lbfgs_data, x, α, β)
    return CompactInverseBFGSOperator{T}(n, n, true, true, prod!, prod!, prod!, compact_lbfgs_data)
end

CompactInverseBFGSOperator(n::Int; kwargs...) = CompactInverseBFGSOperator(Float64, n; kwargs...)

"""
    Simple function that returns a vector containing the dot product between
    x and each columns of matrix mat
"""
function compute_inner_prod!(res, scaling_factor, SR, Y, Q, x)
    res .= x
    SRx = (SR') * x
    res .-= SR * (Y' * x)
    res .-= Y * SRx
    res .*= scaling_factor
    res .+= (SR * (Q * SRx))
end

function compute_indices!(indices, k::Int, m::Int)
    if k <= m
        append!(indices, k)
    else
        for i in 1:m
            indices[i] = (indices[i] % m) + 1
        end
    end
end

"""
    push!(op, s, y)

Push a new {s,y} pair into a compact direct L-BFGS operator.
"""
function push!(
    op::CompactInverseBFGSOperator{T,I,F1,F2,F3},
    s::Vector{T},
    y::Vector{T},
) where {T,I,F1,F2,F3}

    if (y' * s) <= eps(T)
        return op
    end

    (; S, Y, R, Q, SR, scaling, scaling_factor, mem, k, indices) = op.data

    # update iteration number
    k = k + 1

    # Get indices to have right view
    compute_indices!(indices, k, mem)
    iEnd = indices[end]

    # Update S and Y matrices
    S[:, iEnd] .= s
    Y[:, iEnd] .= y

    # Update Lower triangular matrix R
    fill!(view(R, iEnd, :), 0)

    # Update D + scaling_factor * YtY matrix
    for i in indices[1:end-1]
        R[i, iEnd] = dot(y, S[:, i])
        Q[iEnd, i] = scaling_factor * dot(y, Y[:, i])
        Q[i, iEnd] = Q[iEnd, i]
    end

    Q[iEnd, iEnd] = scaling_factor * dot(y, y) + dot(s, y)
    R[iEnd, iEnd] = dot(y, s)

    # Compute S * inv(R)'
    SR = view(S, :, indices) *
         inv(UpperTriangular(view(R, indices, indices)))'

    # Update scaling factor
    if scaling
        scaling_factor = dot(s, y) / dot(y, y)
    end

    op.data.k = k
    op.data.SR = SR
    op.data.indices = indices
    return op
end

