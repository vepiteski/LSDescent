export NwtdirectionLDLt

include("ldlt_symm.jl")

function NwtdirectionLDLt(H,g;verbose::Bool=false)
    L = Array(Float64,2)
    D = Array(Float64,2)
    pp = Array(Int,1)
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

    if true in isnan(D) 
 	println("*******   Problem in D from LDLt: NaN")
        println(" cond (H) = $(cond(H))")
        #res = PDataLDLt()
        #res.OK = false
        #return res
        return (NaN, NaN, NaN, Inf, false, true, :fail)
    end

    Δ, Q = eig(D)

    ϵ2 =  1.0e-8
    Γ = max(abs(Δ),ϵ2)

    # Ad = P'*L*Q*Δ*Q'*L'*Pd =    -g
    # replace Δ by Γ to ensure positive definiteness
    d̃ = L\g[pp]
    d̂ = L'\ (Q*(Q'*d̃ ./ Γ))
    d = - d̂[invperm(pp)]

    return d
end


