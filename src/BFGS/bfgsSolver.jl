function bfgs(nlp     :: AbstractNLPModel;
              x       :: Vector{T}=copy(nlp.meta.x0),
              ϵ       :: T = 1e-6,
              maxiter :: Int = 200,
              scaling :: Bool = true,
              Lp      :: Real = 2, # norm Lp 
              B₀      :: Union{AbstractLinearOperator,
                               AbstractMatrix,
                               UniformScaling{T},
                               Nothing}              = nothing
              ) where T
    
    @info log_header([:iter, :f, :dual, :step, :slope], [Int, T, T, T, T],
                     hdr_override=Dict(:f=>"f(x)", :dual=>"‖∇f‖", :slope=>"∇fᵀd"))
    f = obj(nlp,x)
    ∇f = grad(nlp, x)

    xt = similar(x)
    ∇ft = similar(∇f)

    B = one(eltype(x))*I
    if B₀ != nothing
        B = B₀
    end
    B₀ = B
    #@show typeof(B)
    #@show B
    τ₀ = 0.0005
    τ₁ = 0.9999

    iter = 0
    @info log_row(Any[iter, f, norm(∇f)])  

    while (norm(∇f, Lp) > ϵ) && (iter <= maxiter)
        d = - B*∇f

        #------------------------------------------
        # Hard coded line search
        hp0 = ∇f'*d
        t=1.0
        # Simple Wolfe forward tracking
        xt = x + t*d
        ∇ft = grad(nlp,xt)
        hp = ∇ft'*d
        ft = obj(nlp, xt)
        #  while  ~wolfe & armijo
        nbW = 0
        while (hp <= τ₁ * hp0) && (ft <= ( f + τ₀*t*hp0)) && (nbW < 10)
            t *= 5
            xt = x + t*d
            ∇ft = grad(nlp,xt)
            hp = ∇ft'*d
            ft = obj(nlp, xt)
            nbW += 1
            @debug "W", ft
        end
        tw = t
        
        # Simple Armijo backtracking
        nbk = 0
        while (ft > ( f + τ₀*t*hp0)) && (nbk < 20)
            t *= 0.4
            xt = x + t*d
            ft = obj(nlp, xt)
            nbk += 1
            @debug "A", ft
        end
        #------------------------------------------
        
        if t!=tw   ∇ft = grad(nlp, xt) end
        
        # Update BFGS approximation.
        B = push!(B, t * d, ∇ft - ∇f)    
        #B = push!(B, t * d, ∇ft - ∇f, scaling)    

        #move on
        x .= xt
        f = ft
        ∇f .= ∇ft
        
        iter += 1
        
        @info log_row(Any[iter, ft, norm(∇ft, Lp), t, hp0])
    end
    
    if iter > maxiter
        @warn "Maximum d'itérations"
        #@show B.data.M
    end
    
    return iter, f, norm(∇f, Lp), B, x
end


""" L-BFGS wrapper of the gereral BFGS implementation.
"""
function L_bfgs(nlp     :: AbstractNLPModel;
                x       :: Vector{T}=copy(nlp.meta.x0),
                ϵ       :: T = 1e-6,
                mem     :: Int = 5,
                maxiter :: Int = 200,
                scaling :: Bool = true,
                Lp      :: Real = 2, # norm Lp 
                kwargs...
                ) where T

    @debug "U_Solver = L_bfgs"
    n = nlp.meta.nvar
    B₀ =  InverseLBFGSOperator(Float64, n, mem=mem, scaling=scaling)
    
    return bfgs(nlp; x=x, ϵ = ϵ, maxiter = maxiter, scaling=scaling, Lp = Lp, B₀=B₀, kwargs...)
end


""" M-BFGS wrapper of the gereral BFGS implementation.  M == Matrix
"""
function M_bfgs(nlp     :: AbstractNLPModel;
                x       :: Vector{T}=copy(nlp.meta.x0),
                ϵ       :: T = 1e-6,
                maxiter :: Int = 200,
                scaling :: Bool = true,
                Lp      :: Real = 2, # norm Lp 
                kwargs...
                ) where T

    @debug "U_Solver = M_bfgs"
    n = nlp.meta.nvar
    B₀ =  InverseBFGSOperator(Float64, n, scaling=scaling)
    
    return bfgs(nlp; x=x, ϵ = ϵ, maxiter = maxiter, scaling=scaling, Lp = Lp, B₀=B₀, kwargs...)
end

