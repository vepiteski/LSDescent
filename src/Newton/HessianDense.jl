export hessian_dense

function hessian_dense(nlp, x)
    return Matrix(hess(nlp, x))
    #return Symmetric(Matrix(hess(nlp, x)))
end

