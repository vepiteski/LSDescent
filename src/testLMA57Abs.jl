export NewtonLDLt

H=rand(5,5)
H=H+H'
g=ones(5)

M = Ma57
L = SparseMatrixCSC{Float64,Int32}
D57 = SparseMatrixCSC{Float64,Int32}
pp = Array(Int32,1)
s = Array{Float64}
ρ = Float64
ncomp = Int64

H57 = convert(SparseMatrixCSC{Cdouble,Int32}, H)  #  Hard coded Cdouble
try
    M = Ma57(H,print_level=-1)
    ma57_factorize(M)
catch
    println("*******   Problem in MA57_0")
    res = PDataMA57_0()
    res.OK = false
    return res
end

try
    (L, D57, s, pp) = ma57_get_factors(M)
catch
    println("*******   Problem after MA57_0")
    println(" Cond(H) = $(cond(full(H)))")
    res = PDataMA57_0()
    res.OK = false
    return res
end

#################  Future object BlockDiag operator?
vD1 = diag(D57)       # create internal representation for block diagonal D
vD2 = diag(D57,1)     #

vQ1 = ones(vD1)       # vector representation of orthogonal matrix Q
vQ2 = zeros(vD2)      #
vQ2 = zeros(vD2)      #
vQ2m = zeros(vD2)     #
veig = copy(vD1)      # vector of eigenvalues of D, initialized to diagonal of D
# if D diagonal, nothing more will be computed

i=1;
while i<length(vD1)
    if vD2[i] == 0.0
        i += 1
    else
        mA = [vD1[i] vD2[i];vD2[i] vD1[i+1]]  #  2X2 submatrix
        DiagmA, Qma = eig(mA)                 #  spectral decomposition of mA
        veig[i] = DiagmA[1]
        vQ1[i] = Qma[1,1]
        vQ2[i] = Qma[1,2]
        vQ2m[i] = Qma[2,1]
        vQ1[i+1] = Qma[2,2]
        veig[i+1] = DiagmA[2]
        i += 2
    end  
end

Q = spdiagm((vQ1,vQ2m,vQ2),[0,-1,1])           # sparse representation of Q

Δ = veig

ϵ2 =  1.0e-8
Γ = max(abs(Δ),ϵ2)

# Ad = P'*L*Q*Δ*Q'*L'*Pd =    -g
P=eye(5,5)
P = P[:,pp]
Hr = diagm(1 ./s)*P'*L*Q*diagm(Δ)*Q'*L'*P*diagm(1 ./s)
# replace Δ by Γ to ensure positive definiteness
sg = s .* g

d̃ = L\sg[pp]
d̂ = L'\ (Q*(Q'*d̃ ./ Γ))
d = - d̂[invperm(pp)] .* s
