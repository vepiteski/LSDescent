export Newton

function Newton(nlp :: AbstractNLPModel;
                stp :: TStopping=TStopping(),
                verbose :: Bool=false,
                verboseLS :: Bool = false,
                verboseCG :: Bool = false,
                mem :: Int=5,
                linesearch :: Function = Newarmijo_wolfe,
                Nwtdirection :: Function = NwtdirectionCG,
                hessian_rep :: Function = hessian_operator,
                kwargs...)

    x = copy(nlp.meta.x0)
    n = nlp.meta.nvar

    # xt = Array(Float64, n)
    # ∇ft = Array(Float64, n)
    xt = Array{Float64}(n)
    ∇ft = Array{Float64}(n)

    f = obj(nlp, x)
    #∇f = grad(nlp, x)
    stp, ∇f = start!(nlp,stp,x)
    ∇fNorm = BLAS.nrm2(n, ∇f, 1)

    H = hessian_rep(nlp,x)

    iter = 0


    verbose && @printf("%4s  %8s  %7s  %8s  %4s %8s\n", " iter", "f", "‖∇f‖", "∇f'd", "bk","t")
    verbose && @printf("%5d  %8.1e  %7.1e", iter, f, ∇fNorm)

    optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,x,f,∇f)

    OK = true
    stalled_linesearch = false
    stalled_ascent_dir = false

    β = 0.0
    d = zeros(∇f)
    scale = 1.0

    h_f = 0; h_g = 0; h_h = 0

    while (OK && !(optimal || tired || unbounded))
        d = Nwtdirection(H,∇f,verbose=verboseCG)
        slope = BLAS.dot(n, d, 1, ∇f, 1)

        verbose && @printf("  %8.1e", slope)

        # Perform linesearch.
        if iter < 1
          h = LineModel(nlp, x, d)
        else
          h = Optimize.redirect!(h, x, d)
        end

        verboseLS && println(" ")

        h_f_init = copy(nlp.counters.neval_obj); h_g_init = copy(nlp.counters.neval_grad); h_h_init = copy(nlp.counters.neval_hprod)
        t, t_original, good_grad, ft, nbk, nbW, stalled_linesearch = linesearch(h, f, slope, ∇ft; verboseLS = verboseLS, kwargs...)
        h_f += copy(copy(nlp.counters.neval_obj) - h_f_init); h_g += copy(copy(nlp.counters.neval_grad) - h_g_init); h_h += copy(copy(nlp.counters.neval_hprod) - h_h_init)

        if linesearch in interfaced_algorithms
          ft = obj(nlp, x + (t)*d)
          nlp.counters.neval_obj += -1
        end

        if verboseLS
           (verbose) && print(" \n")
         else
           (verbose) && @printf("  %4d %8s\n", nbk,t)
         end


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

    return (x, f, stp.optimality_residual(∇f), iter, optimal, tired, status, h_f, h_g, h_h)
end
