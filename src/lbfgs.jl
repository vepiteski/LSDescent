export Newlbfgs

function Newlbfgs(nlp :: AbstractNLPModel;
                  stp :: TStopping=TStopping(),
                  verbose :: Bool=false,
                  verboseLS :: Bool = false,
                  mem :: Int=5,
                  linesearch :: Function = Newarmijo_wolfe,
                  kwargs...)

    print_h=false
    x = copy(nlp.meta.x0)
    n = nlp.meta.nvar

    # xt = Array(Float64, n)
    # ∇ft = Array(Float64, n)
    xt = Array{Float64}(n)
    ∇ft = Array{Float64}(n)

    f = obj(nlp, x)
    H = InverseLBFGSOperator(n, mem, scaling=true)

    iter = 0

    #∇f = grad(nlp, x)
    stp, ∇f = start!(nlp,stp,x)
    ∇fNorm = BLAS.nrm2(n, ∇f, 1)

    verbose && @printf("%4s  %8s  %7s  %8s  %4s\n", "iter", "f", "‖∇f‖", "∇f'd", "bk")
    verbose && @printf("%4d  %8.1e  %7.1e", iter, f, ∇fNorm)

    optimal, unbounded, tired, elapsed_time = stop(nlp,stp,iter,x,f,∇f)

    OK = true
    stalled_linesearch = false
    stalled_ascent_dir = false

    d = zeros(∇f)
    h = LineModel(nlp, x, d)

    while (OK && !(optimal || tired || unbounded))
        d = - H * ∇f
        slope = BLAS.dot(n, d, 1, ∇f, 1)
        #slope < 0.0 || error("Not a descent direction! slope = ", slope)
        if slope > 0.0
          stalled_linesearch =true
          verbose && @printf("  %8.1e", slope)
        else
          # Perform linesearch.
          h = Optimize.redirect!(h, x, d)

          debug = false

          t, t_original, good_grad, ft, nbk, nbW, stalled_linesearch = linesearch(h, f, slope, ∇ft; verboseLS = verboseLS, kwargs...)

          if linesearch in interfaced_ls_algorithms
            ft = obj(nlp, x + (t)*d)
            nlp.counters.neval_obj += -1
          end

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

    return (x, f, stp.optimality_residual(∇f), iter, optimal, tired, status, h.counters.neval_obj, h.counters.neval_grad, h.counters.neval_hess)
end
