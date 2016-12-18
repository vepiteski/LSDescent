export CG_generic

function CG_generic(nlp :: AbstractNLPModel;
                    atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
                    max_eval :: Int=0,
                    verbose :: Bool=true,
                    verboseLS :: Bool = true,
                    mem :: Int=5,
                    linesearch :: Function = Newarmijo_wolfe,
                    CG_formula :: Function = formula_HZ,
                    scaling :: Bool = true,
                    kwargs...)

    x = copy(nlp.meta.x0)
    n = nlp.meta.nvar
    
    xt = Array(Float64, n)
    ∇ft = Array(Float64, n)
    
    f = obj(nlp, x)
    ∇f = grad(nlp, x)
    
    ∇fNorm = norm(∇f, Inf)

    ϵ = atol + rtol * ∇fNorm
    max_eval == 0 && (max_eval = max(min(100, 2 * n), 5000))
    iter = 0
    
    verbose && @printf("%4s  %8s  %7s  %8s  %4s\n", "iter", "f", "‖∇f‖", "∇f'd", "bk")
    verbose && @printf("%4d  %8.1e  %7.1e", iter, f, ∇fNorm)
    
    optimal = ∇fNorm <= ϵ
    tired = nlp.counters.neval_obj + nlp.counters.neval_grad > max_eval
    
    β = 0.0
    d = zeros(∇f)
    scale = 1.0
    
    while !(optimal || tired)
        d = - ∇f + β*d
        slope = ∇f⋅d
        if slope > 0.0  # restart with negative gradient
            d = - ∇f
            slope =  ∇f⋅d
        end

        verbose && @printf("  %8.1e", slope)
        
        # Perform improved Armijo linesearch.
        h = C1LineFunction(nlp, x, d*scale)
        t, good_grad, ft, nbk, nbW = linesearch(h, f, slope*scale, ∇ft, verbose=verboseLS; kwargs...)
        t *= scale
        verbose && @printf("  %4d  %8e  %8e \n", nbk, t, scale)
        
        xt = x + t*d

        good_grad || (∇ft = grad!(nlp, xt, ∇ft))
        
        # Move on.
        s = xt - x
        y = ∇ft - ∇f
        β = CG_formula(∇f,∇ft,s,d)

        if scaling
            scale = (y⋅s) / (y⋅y) 
        end
        if scale <= 0.0 
            println(" scale = ",scale)
            println(" ∇f⋅s = ",∇f⋅s,  " ∇ft⋅s = ",∇ft⋅s)
            scale = 1.0
        end
        x = xt
        f = ft
        BLAS.blascopy!(n, ∇ft, 1, ∇f, 1)

        # norm(∇f) bug: https://github.com/JuliaLang/julia/issues/11788
        ∇fNorm = norm(∇f, Inf)
        iter = iter + 1
        
        verbose && @printf("%4d  %8.1e  %7.1e", iter, f, ∇fNorm)
        
        optimal = ∇fNorm <= ϵ
        tired = nlp.counters.neval_obj + nlp.counters.neval_grad > max_eval
    end
    verbose && @printf("\n")
    
    status = tired ? "maximum number of evaluations" : "first-order stationary"
    return (x, f, ∇fNorm, iter, optimal, tired, status)
end
