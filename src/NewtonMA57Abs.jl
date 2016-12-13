export NewtonMA57

function NwtdirectionMA57(H,g)
    M = Ma57
    L = SparseMatrixCSC{Float64,Int32}
    D57 = SparseMatrixCSC{Float64,Int32}
    pp = Array(Int32,1)
    s = Array{Float64}
    ρ = Float64
    ncomp = Int64
    
    H57 = convert(SparseMatrixCSC{Cdouble,Int32}, H)  #  Hard coded Cdouble
    try
        M = Ma57(H,print_level=-1)
        ma57_factorize(M)
    catch
 	println("*******   Problem in MA57_0")
        res = PDataMA57_0()
        res.OK = false
        return res
    end

    try
        (L, D57, s, pp) = ma57_get_factors(M)
    catch
        println("*******   Problem after MA57_0")
        println(" Cond(H) = $(cond(full(H)))")
        res = PDataMA57_0()
        res.OK = false
        return res
    end

    #################  Future object BlockDiag operator?
    vD1 = diag(D57)       # create internal representation for block diagonal D
    vD2 = diag(D57,1)     #

    vQ1 = ones(vD1)       # vector representation of orthogonal matrix Q
    vQ2 = zeros(vD2)      #
    vQ2 = zeros(vD2)      #
    vQ2m = zeros(vD2)     #
    veig = copy(vD1)      # vector of eigenvalues of D, initialized to diagonal of D
                          # if D diagonal, nothing more will be computed
    
    i=1;
    while i<length(vD1)
        if vD2[i] == 0.0
            i += 1
        else
            mA = [vD1[i] vD2[i];vD2[i] vD1[i+1]]  #  2X2 submatrix
            DiagmA, Qma = eig(mA)                 #  spectral decomposition of mA
            veig[i] = DiagmA[1]
            vQ1[i] = Qma[1,1]
            vQ2[i] = Qma[1,2]
            vQ2m[i] = Qma[2,1]
            vQ1[i+1] = Qma[2,2]
            veig[i+1] = DiagmA[2]
            i += 2
        end  
    end

    Q = spdiagm((vQ1,vQ2m,vQ2),[0,-1,1])           # sparse representation of Q
    
    Δ = veig

    ϵ2 =  1.0e-8
    Γ = max(abs(Δ),ϵ2)

    # Ad = P'*L*Q*Δ*Q'*L'*Pd =    -g
    # replace Δ by Γ to ensure positive definiteness
    sg = s .* g

    d̃ = L\sg[pp]
    d̂ = L'\ (Q*(Q'*d̃ ./ Γ))
    d = - d̂[invperm(pp)] .* s

    return d
end

function NewtonMA57(nlp :: AbstractNLPModel;
               atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
               max_eval :: Int=0,
               verbose :: Bool=true,
               mem :: Int=5,
               linesearch :: Function = Newarmijo_wolfe,
               kwargs...)

    x = copy(nlp.meta.x0)
    n = nlp.meta.nvar
    
    xt = Array(Float64, n)
    ∇ft = Array(Float64, n)
    
    f = obj(nlp, x)
    ∇f = grad(nlp, x)

    H = hess(nlp,x)
    H = (H+tril(H,-1)')
    
    ∇fNorm = BLAS.nrm2(n, ∇f, 1)
    ϵ = atol + rtol * ∇fNorm
    max_eval == 0 && (max_eval = max(min(100, 2 * n), 5000))
    iter = 0
    
    verbose && @printf("%4s  %8s  %7s  %8s  %4s\n", "iter", "f", "‖∇f‖", "∇f'd", "bk")
    verbose && @printf("%4d  %8.1e  %7.1e", iter, f, ∇fNorm)
    
    optimal = ∇fNorm <= ϵ
    tired = nlp.counters.neval_obj + nlp.counters.neval_grad + nlp.counters.neval_hess > max_eval
    
    β = 0.0
    d = zeros(∇f)
    scale = 1.0
    
    while !(optimal || tired)
        d = NwtdirectionMA57(H,∇f)
        slope = BLAS.dot(n, d, 1, ∇f, 1)

        verbose && @printf("  %8.1e", slope)
        
        # Perform improved Armijo linesearch.
        h = C1LineFunction(nlp, x, d)


        t, good_grad, ft, nbk, nbW = linesearch(h, f, slope, ∇ft, verbose=false; kwargs...)
        verbose && @printf("  %4d\n", nbk)
        
        BLAS.blascopy!(n, x, 1, xt, 1)
        BLAS.axpy!(n, t, d, 1, xt, 1)
        good_grad || (∇ft = grad!(nlp, xt, ∇ft))
        
        # Move on.
        s = xt - x
        y = ∇ft - ∇f
        β = (∇ft⋅y) / (∇f⋅∇f)
        x = xt
        f = ft

        H = hess(nlp,x)
        H = (H+tril(H,-1)')

        BLAS.blascopy!(n, ∇ft, 1, ∇f, 1)

        # norm(∇f) bug: https://github.com/JuliaLang/julia/issues/11788
        ∇fNorm = BLAS.nrm2(n, ∇f, 1)
        iter = iter + 1
        
        verbose && @printf("%4d  %8.1e  %7.1e", iter, f, ∇fNorm)
        
        optimal = ∇fNorm <= ϵ
        tired = nlp.counters.neval_obj + nlp.counters.neval_grad + nlp.counters.neval_hess > max_eval
    end
    verbose && @printf("\n")
    
    status = tired ? "maximum number of evaluations" : "first-order stationary"
    return (x, f, ∇fNorm, iter, optimal, tired, status)
end
