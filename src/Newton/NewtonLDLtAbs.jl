export NwtdirectionLDLt, NewtonLDLtAbs

include("ldlt_symm.jl")

function NwtdirectionLDLt(H,g; scale_abs :: Bool = true)
    L = Matrix{Float64}
    D = Matrix{Float64}
    pp = Array{Int}
    ρ = Float64
    ncomp = Int64
    
    try
        (L, D, pp, rho, ncomp) = ldlt_symm(H,'r')
    catch
 	println("*******   Problem in LDLt")
        #res = PDataLDLt()
        #res.OK = false
        #return res
        return (NaN, NaN, NaN, Inf, false, true, :fail)
    end

    # A[pp,pp] = P*A*P' =  L*D*L'

    if true in isnan.(D) 
 	println("*******   Problem in D from LDLt: NaN")
        println(" cond (H) = $(cond(H))")
        #res = PDataLDLt()
        #res.OK = false
        #return res
        return (NaN, NaN, NaN, Inf, false, true, :fail)
    end

    Δ, Q = eig(D)

    ϵ2 =  1.0e-8
    if scale_abs
        Γ = max.(abs.(Δ),ϵ2)
    else
        Γ = max.((Δ),ϵ2)
    end
    # Ad = P'*L*Q*Δ*Q'*L'*Pd =    -g
    # replace Δ by Γ to ensure positive definiteness
    d̃ = L\g[pp]
    d̂ = L'\ (Q*(Q'*d̃ ./ Γ))
    d = - d̂[invperm(pp)]

    return d
end


function NewtonLDLtAbsLS(nlp :: AbstractNLPModel;
                         x :: Vector{T}=copy(nlp.meta.x0),
                         stp :: NLPStopping = NLPStopping(nlp,
                                                          NLPAtX(nlp.meta.x0)), 
                         LS_algo   :: Function = bracket,
                         LS_logger :: AbstractLogger = Logging.NullLogger(),
                         kwargs...) where T
    return  Newton_G_StopLS(nlp;
                            x = x,
                            stp = stp,
                            LS_algo = LS_algo,
                            LS_logger = LS_logger,
                            NwtDirection = NwtdirectionSpectral,
                            hessian_rep = hessian_dense,
                            kwargs...)
end

    
