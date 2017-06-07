export NewtonS

function NewtonS(nlp :: AbstractNLPModel;
                 stp :: TStopping=TStopping(),
                 verbose :: Bool=false,
                 verboseLS :: Bool = false,
                 verboseCG :: Bool = false,
                 mem :: Int=5,
                 linesearch :: Function = Newarmijo_wolfe,
                 Nwtdirection :: Function = NwtdirectionCG,
                 hessian_rep :: Function = hessian_dense,
                 kwargs...)

    x = copy(nlp.meta.x0)
    n = nlp.meta.nvar

    xt = Array(Float64, n)
    ∇ft = Array(Float64, n)

    f = obj(nlp, x)
    ∇f = grad(nlp, x)

    H = hessian_rep(nlp,x)

    iter = 0

    stp = start!(nlp,stp,x)

    verbose && @printf("%4s  %8s  %7s  %8s  %4s\n", "iter", "f", "‖∇f‖", "∇f'd", "bk")
    verbose && @printf("%4d  %8.1e  %7.1e", iter, f, ∇fNorm)

    optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,x,f,∇f)

    OK = true
    stalled_linesearch = false
    stalled_ascent_dir = false

    β = 0.0
    d = zeros(∇f)
    scale = 1.0

    while (OK && !(optimal || tired || unbounded))
        d = Nwtdirection(H,∇f,verbose=verboseCG)
        slope = BLAS.dot(n, d, 1, ∇f, 1)

        verbose && @printf("  %8.1e", slope)

        # Perform improved Armijo linesearch.
        # if linesearch==TR_Nwt_ls || linesearch==ARC_Nwt_ls || linesearch==trouve_intervalleA_ls
        #    h = C2LineFunction(nlp, x, d)
        #   t, good_grad, ft, nbk, nbW = linesearch(h, f, slope, ∇ft, verbose=verboseLS; kwargs...)
        # else
        h = C1LineFunction(nlp, x, d)
        t, good_grad, ft, nbk, nbW = linesearch(h, f, slope, ∇ft, verbose=verboseLS; kwargs...)
        #end

        #t, good_grad, ft, nbk, nbW = linesearch(h, f, slope, ∇ft, verbose=false; kwargs...)

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

        H = hessian_rep(nlp,x)

        BLAS.blascopy!(n, ∇ft, 1, ∇f, 1)

        # norm(∇f) bug: https://github.com/JuliaLang/julia/issues/11788
        ∇fNorm = BLAS.nrm2(n, ∇f, 1)
        iter = iter + 1

        verbose && @printf("%4d  %8.1e  %7.1e", iter, f, ∇fNorm)
        optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,x,f,∇f)
        OK = !stalled_linesearch & !stalled_ascent_dir

    end

    verbose && @printf("\n")

    if optimal status = :Optimal
    elseif unbounded status = :Unbounded
    elseif stalled_linesearch status = :StalledLinesearch
    elseif stalled_ascent_dir status = :StalledAscentDir
    else status = :UserLimit
    end

    return (x, f, stp.optimality_residual(∇f), iter, optimal, tired, status)
end
