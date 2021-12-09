export hessian_sparse

using SparseArrays

function hessian_sparse(nlp, x)
    #Hx = hess(nlp,x)
    #H = Symmetric(Hx, :L)
    return (hess(nlp, x))
end

