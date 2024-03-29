
""" Accept any form of initial quasi-Newton representation
- LinearOperator (should be a quasiNewtonOperator)
- Matrix
- I
- nothing

    defaults to InverseBFGSOperator   initialized to I
"""
function AcceptAll(T, n,
                   B₀    :: Union{AbstractLinearOperator,
                                  AbstractMatrix,
                                  UniformScaling,
                                  Nothing}              = nothing,
                   )

    # default
    B = InverseBFGSOperator(T, n)

    if B₀ != nothing
        if isa(B₀, AbstractMatrix)
            B = InverseBFGSOperator(B₀)
        elseif isa(B₀, AbstractLinearOperator)
            B = B₀
        else @warn "Unsupported quasiNewton Operator, using InverseBFGSOperator"
        end
    end

    return B
end
