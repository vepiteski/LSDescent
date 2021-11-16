function Newton_Spectral(nlp     :: AbstractNLPModel;
                         x       :: AbstractVector=copy(nlp.meta.x0),
                         ϵ       :: Real=√eps(eltype(x)),
                         #ϵ       :: T = 1e-6,
                         maxiter :: Int = 200,
                         Lp      :: Real = 2 # norm Lp 
                         )# where T<:Real

    T = eltype(x)
    n = nlp.meta.nvar
    @info log_header([:iter, :f, :dual, :step, :slope], [Int, T, T, T, T],
                     hdr_override=Dict(:f=>"f(x)", :dual=>"‖∇f‖", :slope=>"∇fᵀd"))
    f  ::T = obj(nlp,x)
    ∇f :: Vector{T} = grad(nlp, x)
    
    τ₀ = 0.0005
    τ₁ = 0.999

    iter = 0
    @info log_row(Any[iter, f, norm(∇f, Lp)])  

    while (norm(∇f, Lp) > ϵ) && (iter <= maxiter)
        H = Matrix(Symmetric(hess(nlp, x),:L))
        Δ, O = eigen(H)
        # Boost negative values of Δ to 1e-8
        # devise an adaptative value for γ
        γ = 1e-6
        D = abs.(Δ) + max.((γ .- abs.(Δ)), 0.0) .*ones(n)
        
        d = - O*diagm(1.0 ./ D)*O'*∇f

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
            t *= 0.5
            xt = x + t*d
            ft = obj(nlp, xt)
            nbk += 1
            @debug "A", ft
        end
        #------------------------------------------
        
        x += t*d
        f = ft
        if t!=tw   ∇ft = grad(nlp, xt) end
        ∇f = ∇ft
        iter += 1
        
        @info log_row(Any[iter, f, norm(∇f, Lp), t, hp0])
    end
    if iter > maxiter @warn "Maximum d'itérations"
    end
    
    return iter, f, norm(∇f, Lp), x
end
