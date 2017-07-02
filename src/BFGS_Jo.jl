# using JuMP
# using NLPModels
# using Optimize
# using LinearOperators
# include("scosine.jl");
#
# include("armijo_wolfe.jl")
# include("arwheadJo.jl");
# prob1=MathProgNLPModel(arwheadJo());
#
# (xₖ,f,gₖNorm,iter)=BFGS_Jo(prob1)

export BFGS_Jo_O

function BFGS_Jo_O(nlp :: AbstractNLPModel;
                   atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
                   max_eval :: Int=0,
                   ϵ :: Float64=1.0e-5,
                   linesearch :: Function = Newarmijo_wolfe,
                   kwargs...)

  x = copy(nlp.meta.x0)  #valeur initiale
  n = nlp.meta.nvar  #nombre de variable

  xt = Array(Float64, n)  #Initialisation
  yₖ = Array(Float64, n)
  ∇ft = Array(Float64, n)

  f = obj(nlp, x)  #valeur de depart
  ∇f = grad(nlp, x)
  H = eye(n,n)

  ∇fNorm = BLAS.nrm2(n, ∇f, 1)
  # ϵ = atol + rtol * ∇fNorm  #fabrication des conditions d'arret
  max_eval == 0 && (max_eval = max(min(100, 2 * n), 5000))
  iter = 0

  optimal = ∇fNorm <= ϵ
  tired = nlp.counters.neval_obj + nlp.counters.neval_grad > max_eval

  h_f = 0; h_g = 0; h_h = 0

  while !(optimal || tired)

    #println("H = $H")
    d  = -H * ∇f  #calcul de la direction
    #println("dₖₚ = $d")
    slope = BLAS.dot(n, d, 1, ∇f, 1)  #calcul du pa
    #println("slope = $slope")
    slope < 0.0 || error("Not a descent direction! slope = ", slope)

    # Perform improved Armijo linesearch.
    h = C1LineFunction2(nlp, x, d)
    t, good_grad, ft, nbk, nbW, stalled_linesearch, h_f_c, h_g_c, h_h_c = linesearch(h, f, slope, ∇ft;kwargs...)
    h_f += h_f_c; h_g += h_g_c; h_h += h_h_c
    #println("t = $t")
    #println("x = $x")
    #println("xt = $xt")

    xt = copy(x)
    xt = xt + t * d
    # BLAS.blascopy!(n, copy(x), 1, xt, 1)
    # BLAS.axpy!(n, t, d, 1, xt, 1)

    #println("x = $x")
    #println("xt = $xt")
    # if iter == 2
    #   error()
    # end
    #println("∇ft = $∇ft")
    good_grad || (∇ft = grad!(nlp, xt, ∇ft))
    #println("∇ft = $∇ft")
    # Update BFGS approximation.
    yₖ  = ∇ft - ∇f
    #println("yₖ = $yₖ")
    sₖ  = xt - x
    #println("sₖ = $sₖ")
    ρₖ  = 1 / (dot(vec(yₖ),vec(sₖ)) )
    #println("ρₖ = $ρₖ")
    Vₖ  = eye(n,n) - (ρₖ * (yₖ * (sₖ)'))
    #if iter==0
      H = (dot(yₖ,sₖ)/dot(H*yₖ,yₖ)) * H
    #end
    Hₖ   = ((Vₖ)' * H * Vₖ) + ρₖ*(sₖ * (sₖ)')
    # println(H * ones(n))
    # SIGN = sign(dot(vec(yₖ),vec(sₖ))) == 1
    # Hpsitive_def = isposdef(Hₖ)
    # println("SIGN = ",SIGN == Hpsitive_def)
    # if Hpsitive_def
    #   print(Hpsitive_def)
    #   H=Hₖ
    # end
    H=Hₖ
    hhh=Hₖ * ones(n)
    #print_with_color(:cyan,string(hhh))
    #println("H = $Hₖ")
    #print("H = $H")
    #println(" ")
    #println(H * ones(n))
    #return H
    # Move on.
    x = xt
    BLAS.blascopy!(n, ∇ft, 1, ∇f, 1)

    ∇fNorm = BLAS.nrm2(n, ∇f, 1)
    iter = iter + 1

    optimal = ∇fNorm <= ϵ
    tired = nlp.counters.neval_obj + nlp.counters.neval_grad > max_eval

    # if iter==3
    #   error("done")
    # end
  end
  return (x, f, ∇fNorm, iter, optimal, tired, optimal, h_f, h_g, h_h)
end

# julia> (x, f, ∇fNorm, iter, optimal, tired, status)=Newlbfgs(prob1)
# ([1.0,1.0,1.0,4.79305e-8],2.4780177909633494e-13,2.4421831884985273e-6,8,true,false,"first-order stationary")
