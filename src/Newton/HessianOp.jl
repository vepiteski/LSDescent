export hessian_operator

function hessian_operator(nlp, x::Vector{T}) where T
    n = nlp.meta.nvar
    temp = similar(x)
    return hess_op!(nlp, x, temp)
end
