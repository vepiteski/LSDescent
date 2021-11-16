export CG_generic

function CG_generic(nlp :: AbstractNLPModel;
                    stp :: TStopping = TStopping(),
                    verbose :: Bool=false,
                    verboseLS :: Bool = false,
                    linesearch :: Function = Newarmijo_wolfe,
                    CG_formula :: Function = formula_HZ,
                    scaling :: Bool = true,
                    print_h :: Bool = false,
                    print_h_iter :: Int64 = 1,
                    kwargs...)

    x = copy(nlp.meta.x0)
    n = nlp.meta.nvar

    xt = Array{Float64}(n)
    ∇ft = Array{Float64}(n)

    f = obj(nlp, x)

    iter = 0

    #∇f = grad(nlp, x)
    stp, ∇f = start!(nlp,stp,x)
    ∇fNorm = BLAS.nrm2(n, ∇f, 1)

    if n <= 2
      verbose && @printf("%4s  %8s  %11s %17s  %13s     %4s  %2s   %14s  %14s  %14s \n", "iter", "f", "‖∇f‖", "x", "∇f'd", "bk","t","scale","h'(t)","t_original")
      verbose && @printf("%4d  %8e  %7.1e %24s", iter, f, ∇fNorm,x)
    else
      verbose && @printf("%4s  %8s  %11s %8s     %4s  %2s   %14s  %14s  %14s \n", "iter", "f", "‖∇f‖", "∇f'd", "bk","t","scale","h'(t)","t_original")
      verbose && @printf("%4d  %8e  %7.1e", iter, f, ∇fNorm)
    end

    optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,x,f,∇f)

    β = 0.0
    d = zeros(∇f)
    scale = 1.0

    h = LineModel(nlp, x, d*scale)

    OK = true
    stalled_linesearch = false
    stalled_ascent_dir = false

    while (OK && !(optimal || tired || unbounded) )
        d = - ∇f + β*d
        slope = ∇f⋅d
        if slope > 0.0  # restart with negative gradient
          d = - ∇f
          slope =  ∇f⋅d
        end
        verbose && @printf(" %10.1e", slope*scale)

        h = Optimize.redirect!(h, x, d*scale)

        debug = false

        verboseLS && println(" ")

        t, t_original, good_grad, ft, nbk, nbW, stalled_linesearch = linesearch(h, f, slope * scale, ∇ft; verboseLS = verboseLS, debug = debug, kwargs...)

        t *= scale
        if verboseLS
          verbose && print("\n")
        else
          verbose && @printf("  %4d  %8e  %8e %8e  %8e\n", nbk, t, scale,grad(h,t),t_original*scale)
        end

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
            scale = 1.0
        end
        x = xt
        f = ft
        BLAS.blascopy!(n, ∇ft, 1, ∇f, 1)

        # norm(∇f) bug: https://github.com/JuliaLang/julia/issues/11788
        ∇fNorm = norm(∇f, Inf)
        iter = iter + 1


        if n <= 2
          verbose && @printf("%4d  %8e  %7.1e %24s", iter, f, ∇fNorm, x)
        else
          verbose && @printf("%4d  %8e  %7.1e", iter, f, ∇fNorm)
        end


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

    return (x, f, stp.optimality_residual(∇f), iter, optimal, tired, status, h.counters.neval_obj, h.counters.neval_grad, h.counters.neval_hess)
end
