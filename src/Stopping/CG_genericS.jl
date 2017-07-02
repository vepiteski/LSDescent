export CG_genericS

function CG_genericS(nlp :: AbstractNLPModel;
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

    xt = Array(Float64, n)
    ∇ft = Array(Float64, n)

    f = obj(nlp, x)
    ∇f = grad(nlp, x)
    ∇fNorm = BLAS.nrm2(n, ∇f, 1)

    iter = 0

    stp = start!(nlp,stp,x)

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

    OK = true
    stalled_linesearch = false
    stalled_ascent_dir = false

    h_f = 0; h_g = 0; h_h = 0

    while (OK && !(optimal || tired || unbounded) )
        d = - ∇f + β*d
        slope = ∇f⋅d
        if slope > 0.0  # restart with negative gradient
          #stalled_ascent_dir = true
          d = - ∇f
          slope =  ∇f⋅d
        end
        # #else
        verbose && @printf(" %10.1e", slope*scale)

        # Perform improved Armijo linesearch.
        if linesearch in Newton_linesearch
          h = C2LineFunction2(nlp, x, d*scale)
        else
          h = C1LineFunction2(nlp, x, d*scale)
        end

        debug = false

        if print_h && (iter == print_h_iter)
          debug= true
          graph_linefunc(h, f, slope*scale;kwargs...)
        end

        verboseLS && println(" ")

        if linesearch in interfaced_algorithms
          h_f_init = copy(nlp.counters.neval_obj); h_g_init = copy(nlp.counters.neval_grad); h_h_init = copy(nlp.counters.neval_hprod)
          t,t_original, good_grad, nbk, nbW, stalled_linesearch = linesearch(h, f, slope*scale, ∇ft; kwargs...)
          h_f += copy(copy(nlp.counters.neval_obj) - h_f_init); h_g += copy(copy(nlp.counters.neval_grad) - h_g_init); h_h += copy(copy(nlp.counters.neval_hprod) - h_h_init)
          ft = obj(nlp, x + (t*scale)*d)
          nlp.counters.neval_obj += -1
        else
          t, t_original, good_grad, ft, nbk, nbW, stalled_linesearch, h_f_c, h_g_c, h_h_c = linesearch(h, f, slope * scale, ∇ft,
                                                                                                       verboseLS = verboseLS,
                                                                                                       debug = debug,
                                                                                                       kwargs...)
          h_f += h_f_c
          h_g += h_g_c
          h_h += h_h_c
        end

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

    return (x, f, stp.optimality_residual(∇f), iter, optimal, tired, status, h_f, h_g, h_h)
end
