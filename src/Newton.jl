export Newton

function Newton(nlp :: AbstractNLPModel;
                atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
                max_eval :: Int=5000,
                max_iter :: Int=20000,
                verbose :: Bool=false,
                verboseLS :: Bool = false,
                verboseCG :: Bool = false,
                mem :: Int=5,
                linesearch :: Function = Newarmijo_wolfe,
                Nwtdirection :: Function = NwtdirectionCG,
                hessian_rep :: Function = hessian_sparse,
                kwargs...)

    x = copy(nlp.meta.x0)
    n = nlp.meta.nvar

    xt = Array(Float64, n)
    ∇ft = Array(Float64, n)

    f = obj(nlp, x)
    ∇f = grad(nlp, x)

    H = hessian_rep(nlp,x)

    ∇fNorm = BLAS.nrm2(n, ∇f, 1)
    ϵ = atol + rtol * ∇fNorm
    iter = 0

    verbose && @printf("%4s  %8s  %7s  %8s  %4s\n", "iter", "f", "‖∇f‖", "∇f'd", "bk")
    verbose && @printf("%4d  %8.1e  %7.1e", iter, f, ∇fNorm)

    optimal = ∇fNorm <= ϵ
    total_calls = nlp.counters.neval_obj + nlp.counters.neval_grad + n*nlp.counters.neval_hess+ nlp.counters.neval_hprod
    #tired = total_calls > max_eval
    tired = iter > max_iter
    β = 0.0
    d = zeros(∇f)
    scale = 1.0

    while !(optimal || tired)
        d = Nwtdirection(H,∇f,verbose=verboseCG)
        slope = BLAS.dot(n, d, 1, ∇f, 1)

        verbose && @printf("  %8.1e", slope)

        # Perform improved Armijo linesearch.
        h = C1LineFunction(nlp, x, d)
        if linesearch==_strongwolfe2!
          x_out = copy(x)
          x_new = copy(x)
          if iter == 0
            gr_new = zeros(n)
          else
            gr_new = ∇f
          end
          p = d
          lsr = LineSearchResults{Float64}([0.0],[f],[slope],0)
          alpha0 = 1.0
          mayterminate = false
          t  = linesearch(nlp, x, p, x_new, lsr, alpha0, mayterminate; kwargs...)
          good_grad = false
          ft = obj(nlp, x + t*d)
          nbk = NaN
          nbW = NaN
        elseif linesearch==_morethuente2!
          x_out = copy(x)
          s = d
          x_new = copy(x)
          lsr = LineSearchResults{Float64}([0.0],[f],[slope],0)
          stp = 1.0
          mayterminate = false
          t= linesearch(nlp, x_out, s, x_new, lsr, stp, mayterminate; kwargs...)
          good_grad = false
          ft = obj(nlp, x + t*d)
          nbk = NaN
          nbW = NaN
        elseif linesearch==_hagerzhang2!
          x_out = copy(x)
          s = d
          xtmp = copy(x)
          lsr = LineSearchResults([0.0],[f],[slope],0)
          c = 1.0
          mayterminate = false
          t = linesearch(nlp, x_out, s, xtmp, lsr, c, mayterminate; kwargs... )
          good_grad = false
          ft = obj(nlp, x + t*d)
          nbk = NaN
          nbW = NaN
        elseif linesearch==_backtracking2!
          x_out = copy(x)
          s = d
          x_scratch = copy(x)
          lsr = LineSearchResults{Float64}([0.0],[f],[slope],0)
          t = linesearch(nlp, x_out, s, x_scratch, lsr; kwargs...)
          good_grad = false
          ft = obj(nlp, x + t*d)
          nbk = NaN
          nbW = NaN
        else
          t, good_grad, ft, nbk, nbW = linesearch(h, f, slope, ∇ft, verbose=false; kwargs...)
        end

        verbose && @printf("  %4d\n", nbk)

        #if linesearch != morethuente2
        BLAS.blascopy!(n, x, 1, xt, 1)
        BLAS.axpy!(n, t, d, 1, xt, 1)
        good_grad || (∇ft = grad!(nlp, xt, ∇ft))
        #end

        # Move on.
        s = xt - x
        y = ∇ft - ∇f
        β = (∇ft⋅y) / (∇f⋅∇f)
        x = xt
        f = ft

        H = hessian_rep(nlp,x)

        BLAS.blascopy!(n, ∇ft, 1, ∇f, 1)

        # norm(∇f) bug: https://github.com/JuliaLang/julia/issues/11788
        ∇fNorm = BLAS.nrm2(n, ∇f, 1)
        iter = iter + 1

        verbose && @printf("%4d  %8.1e  %7.1e", iter, f, ∇fNorm)

        optimal = (∇fNorm <= ϵ) | (isinf(f) & (f<0.0))
        total_calls = nlp.counters.neval_obj + nlp.counters.neval_grad + nlp.counters.neval_hess+ nlp.counters.neval_hprod
        # tired = total_calls > max_eval
        tired = iter > max_iter
    end
    verbose && @printf("\n")

    status = tired ? "UserLimit" : "Optimal"
    return (x, f, ∇fNorm, iter, optimal, tired, status)
end
