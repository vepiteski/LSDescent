export NewlbfgsS

function NewlbfgsS(nlp :: AbstractNLPModel;
                  stp :: TStopping=TStopping(),
                  verbose :: Bool=false,
                  verboseLS :: Bool = false,
                  mem :: Int=5,
                  linesearch :: Function = Newarmijo_wolfe,
                  print_h :: Bool = false,
                  print_h_iter :: Int64 = 1,
                  kwargs...)
print_h=false
    x = copy(nlp.meta.x0)
    n = nlp.meta.nvar

    xt = Array(Float64, n)
    ∇ft = Array(Float64, n)

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

    h_f = 0; h_g = 0; h_h = 0

    while (OK && !(optimal || tired || unbounded))
        d = - H * ∇f
        slope = BLAS.dot(n, d, 1, ∇f, 1)
        #slope < 0.0 || error("Not a descent direction! slope = ", slope)
        if slope > 0.0
          stalled_linesearch =true
          verbose && @printf("  %8.1e", slope)
        else
          # Perform improved Armijo linesearch.
          if linesearch in Newton_linesearch
            h = C2LineFunction2(nlp, x, d)
          else
            h = C1LineFunction2(nlp, x, d)
          end

          debug = false

          if print_h && (iter == print_h_iter)
            debug= true
            graph_linefunc(h, f, slope*scale;kwargs...)
          end

          if linesearch in interfaced_algorithms
            h_f_init = copy(nlp.counters.neval_obj); h_g_init = copy(nlp.counters.neval_grad); h_h_init = copy(nlp.counters.neval_hprod)
            t,t_original, good_grad, nbk, nbW, stalled_linesearch = linesearch(h, f, slope, ∇ft; kwargs...)
            h_f += copy(copy(nlp.counters.neval_obj) - h_f_init); h_g += copy(copy(nlp.counters.neval_grad) - h_g_init); h_h += copy(copy(nlp.counters.neval_hprod) - h_h_init)
            ft = obj(nlp, x + t*d)
            nlp.counters.neval_obj += -1
          else
            t, t_original, good_grad, ft, nbk, nbW, stalled_linesearch, h_f_c, h_g_c, h_h_c = linesearch(h, f, slope, ∇ft,
                                                                                                         verboseLS = verboseLS,
                                                                                                         debug = debug;
                                                                                                         kwargs...)
            h_f += h_f_c
            h_g += h_g_c
            h_h += h_h_c
          end
        #t, good_grad, ft, nbk, nbW = linesearch(h, f, slope, ∇ft, verbose=verboseLS; kwargs...)

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

    return (x, f, stp.optimality_residual(∇f), iter, optimal, tired, status, h_f, h_g, h_h)
end
