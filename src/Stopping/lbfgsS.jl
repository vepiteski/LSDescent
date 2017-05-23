export NewlbfgsS

function NewlbfgsS(nlp :: AbstractNLPModel;
                  stp :: TStopping=TStopping(),
                  verbose :: Bool=false,
                  verboseLS :: Bool = false,
                  mem :: Int=5,
                  linesearch :: Function = Newarmijo_wolfe,
                  kwargs...)

    x = copy(nlp.meta.x0)
    n = nlp.meta.nvar

    xt = Array(Float64, n)
    ∇ft = Array(Float64, n)

    f = obj(nlp, x)
    ∇f = grad(nlp, x)
    H = InverseLBFGSOperator(n, mem, scaling=true)

    iter = 0

    stp = start!(nlp,stp,x)

    verbose && @printf("%4s  %8s  %7s  %8s  %4s\n", "iter", "f", "‖∇f‖", "∇f'd", "bk")
    verbose && @printf("%4d  %8.1e  %7.1e", iter, f, ∇fNorm)

    optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,x,f,∇f)

    OK = true
    stalled_linesearch = false
    stalled_ascent_dir = false

    while (OK && !(optimal || tired || unbounded))
        d = - H * ∇f
        slope = BLAS.dot(n, d, 1, ∇f, 1)
        #slope < 0.0 || error("Not a descent direction! slope = ", slope)
        if slope > 0.0
          stalled_linesearch =true
          verbose && @printf("  %8.1e", slope)
        else
          # Perform improved Armijo linesearch.
          h = C1LineFunction(nlp, x, d)
          t, good_grad, ft, nbk, nbW,stalled_linesearch = linesearch(h, f, slope, ∇ft, verbose=verboseLS; kwargs...)

          verbose && @printf("  %4d\n", nbk)

          BLAS.blascopy!(n, x, 1, xt, 1)
          BLAS.axpy!(n, t, d, 1, xt, 1)
          good_grad || (∇ft = grad!(nlp, xt, ∇ft))

          # Update L-BFGS approximation.
          push!(H, t * d, ∇ft - ∇f)

          # Move on.
          x = xt
          f = ft
          BLAS.blascopy!(n, ∇ft, 1, ∇f, 1)
          # norm(∇f) bug: https://github.com/JuliaLang/julia/issues/11788
          ∇fNorm = BLAS.nrm2(n, ∇f, 1)
          iter = iter + 1

          verbose && @printf("%4d  %8.1e  %7.1e", iter, f, ∇fNorm)

          optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,x,f,∇f)
        end
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
