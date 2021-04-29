
function Newton_Stop(nlp :: AbstractNLPModel;
                     x   :: Vector{T}=copy(nlp.meta.x0),
                     stp :: NLPStopping = NLPStopping(nlp,
                                                      NLPAtX(nlp.meta.x0)),
                     ) where T

    n = nlp.meta.nvar
    @info log_header([:iter, :f, :dual, :step, :slope], [Int, T, T, T, T],
                     hdr_override=Dict(:f=>"f(x)", :dual=>"‖∇f‖", :slope=>"∇fᵀd"))
    f = obj(nlp,x)
    ∇f = grad(nlp, x)

    OK = update_and_start!(stp, x = x, fx = f, gx = ∇f)
    
    τ₀ = 0.0005
    τ₁ = 0.9999

    @info log_row(Any[0, f, norm(∇f)])

    while !OK
        Hx = hess(nlp, x)
        H = Matrix(Symmetric(Hx,:L))
        Stopping.update!(stp.current_state; Hx = Hx)
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

        OK = update_and_stop!(stp, x = x, gx = ∇f, fx = f)

        @info log_row(Any[stp.meta.nb_of_stop, f, norm(∇f), t, hp0])
    end
    
    if !stp.meta.optimal
        #@warn "Optimalité non atteinte"
        @warn status(stp,list=true)
    end
    
    return stp
    #return f, norm(∇f), stp
end
