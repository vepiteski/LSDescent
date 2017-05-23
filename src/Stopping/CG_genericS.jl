export CG_genericS

function CG_genericS(nlp :: AbstractNLPModel;
                     stp :: TStopping = TStopping(),
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

    iter = 0

    stp = start!(nlp,stp,x)

    verbose && @printf("%4s  %8s  %7s  %8s  %4s\n", "iter", "f", "‖∇f‖", "∇f'd", "bk")
    verbose && @printf("%4d  %8.1e  %7.1e", iter, f, ∇fNorm)

    optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,x,f,∇f)

    β = 0.0
    d = zeros(∇f)
    scale = 1.0

    OK = true
    stalled_linesearch = false
    stalled_ascent_dir = false

    while (OK && !(optimal || tired || unbounded) )
        d = - ∇f + β*d
        slope = ∇f⋅d
        if slope > 0.0  # restart with negative gradient
          #stalled_ascent_dir = true
          d = - ∇f
          slope =  ∇f⋅d
        end
        #else
        verbose && @printf("  %8.1e", slope)

        # Perform improved Armijo linesearch.
        h = C1LineFunction(nlp, x, d*scale)
        t, good_grad, ft, nbk, nbW, stalled_linesearch = linesearch(h, f, slope*scale, ∇ft, verbose=verboseLS; kwargs...)
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

    return (x, f, stp.optimality_residual(∇f), iter, optimal, tired, status,elapsed_time)
end
