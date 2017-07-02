export CG_generic

function CG_generic(nlp :: AbstractNLPModel;
                    atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
                    max_eval :: Int=5000,
                    max_iter :: Int=20000,
                    verbose :: Bool=false,
                    verboseLS :: Bool = false,
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
        if linesearch in Newton_linesearch
          h = C2LineFunction(nlp, x, d*scale)
        else
          h = C1LineFunction(nlp, x, d*scale)
        end
        t, good_grad, ft, nbk, nbW = linesearch(h, f, slope*scale, ∇ft, verbose=verboseLS; kwargs...)
        t *= scale
        verbose && @printf("  %4d  %8e  %8e \n", nbk, t, scale)

        xt = x + t*d

        good_grad || (∇ft = grad!(nlp, xt, ∇ft))
        # Move on.
        s = xt - x
        y = ∇ft - ∇f
        β = 0.0
        if (∇ft⋅∇f) < 0.2 * (∇ft⋅∇ft)   # Powell restart
            β = CG_formula(∇f,∇ft,s,d)
        end
        if scaling
            scale = (y⋅s) / (y⋅y)
        end
        if scale <= 0.0
            #println(" scale = ",scale)
            #println(" ∇f⋅s = ",∇f⋅s,  " ∇ft⋅s = ",∇ft⋅s)
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
