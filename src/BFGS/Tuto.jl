# Tutorial to add new quasi-Newton formulas to LSDescent
#
# Use DFP full matrix formula as an illustration. No new code
# will be required since the DFP formula for direct update is exactly
# the same as BFGS formula for inverse update.
#
# We cheat and use the direct DFP update, and thereafter 


using FastClosures

mutable struct QN_Matrix_Data{T}
    M :: Matrix{T}
    scaling::Bool
    Ax :: Vector{T}  # pre allocated for mul5
end

function QN_Matrix_Data(
    M₀ :: Matrix{T};
    scaling :: Bool = true ) where {T}

    n,  = size(M₀) # add checks for square symmetric matrix
    
    QN_Matrix_Data{T}(M₀, scaling, Vector{T}(undef, n))
end


"A type for DFP ."
mutable struct DFPOperator{T, N <: Integer, F, Ft, Fct} <: AbstractLinearOperator{T}
    nrow::N
    ncol::N
    symmetric::Bool
    hermitian::Bool
    prod!::F    # apply the operator to a vector
    tprod!::Ft    # apply the transpose operator to a vector
    ctprod!::Fct   # apply the transpose conjugate operator to a vector
    #inverse::Bool
    data::QN_Matrix_Data{T}
    nprod::N
    ntprod::N
    nctprod::N
end

DFPOperator{T}(
    nrow :: N,
    ncol :: N,
    symmetric :: Bool,
    hermitian :: Bool,
    prod!::F,
    tprod!::Ft,
    ctprod!::Fct,
    data::QN_Matrix_Data{T},
) where {T, N, F, Ft, Fct} = DFPOperator{T, N, F, Ft, Fct}(
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
has_args5(op::DFPOperator) = true
use_prod5!(op::DFPOperator) = true
isallocated5(op::DFPOperator) = true


quasiNewtonOp = Union{quasiNewtonOp, DFPOperator}

"""
    InverseBFGSOperator(M₀, n [; scaling=true])
    InverseBFGSOperator(n, [; scaling=true])
Construct a BFGS approximation in inverse form. 
"""
function InverseDFPOperator(M :: Matrix{T}; kwargs...) where {T <: Real}
    kwargs = Dict(kwargs)
    dfp_data = QN_Matrix_Data(M; kwargs...)

    function dfp_multiply(res::AbstractVector,
                           data::QN_Matrix_Data,
                           x::AbstractArray,
                           αm,
                           βm::T2,
                           ) where T2

        q = data.Ax  # pre allocated
        q .= data.M \ x  #  CHEAT  direct update inversed here

        # mul5 stuff
        if βm == zero(T2)
            res .= αm .* q
        else
            res .= αm .* q .+ βm .* res
        end

        return res
    end
    
    prod! = @closure (res, x, α, β) -> dfp_multiply(res, dfp_data, x, α, β)
    return DFPOperator{T}(n, n, true, true, prod!, prod!, prod!, dfp_data)
end

function InverseDFPOperator(T, n :: N; kwargs...) where {N <: Integer}
    Eye = Matrix{T}(I,n,n)
    InverseDFPOperator(Eye; kwargs...)
end


import Base.push!

function push!(Bop :: DFPOperator{T, N, F1, F2, F3},
               sk:: Vector{T},
               yk:: Vector{T}) where {T, N, F1, F2, F3}

    # simple CHEAT for testing, interchange yk and sk from
    # the inverseBFGS version to get directDFP
    
    scaling = false # Bop.data.scaling
    denom = yk'*sk

    # Very inefficient, compute inv(update_BFGS(inv(B)) using sk <-> yk
    #
    B = inv(Bop.data.M)
    if (denom > 1.0e-20) 
        #self-scaled version aka Luenberger

        #divide vector yk by denom 
        Bs = B*(sk/denom)

        stBs = (sk'*Bs)
        # refactor to yield O(n²) complexity... Much more efficient!
        Bsyt = Bs*yk'

        #divide vector sk by denom
        yyt = (yk/denom)*yk'

        #B = γ * (B - (Byst' + Byst)/denom + (ytBy/denom)*sst/denom)  + sst/denom
        # avoids dividing matrices by denom
        #
        B = (B  - (Bsyt' + Bsyt) + stBs*yyt)  + yyt
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
    #  return the INVERSE of the DIRECT update using sₖ <-> yₖ
    Bop.data.M .= inv(B) 
    
    return Bop
end






#function Matrix(Op::InverseBFGSOperator)
#    return Op.data.M
#end
