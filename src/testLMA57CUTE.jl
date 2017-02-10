using HSL
using CUTEst

nlp = CUTEstModel("PALMER8C")
x = copy(nlp.meta.x0)
n = nlp.meta.nvar
H = hess(nlp,x)
H = (H+tril(H,-1)')
finalize(nlp)
run(`rm AUTOMAT.d`)
run(`rm OUTSDIF.d`)
run(`rm libPALMER8C.so`)

H2=rand(n,n)
H2=sparse(H2+H2')

H = convert(SparseMatrixCSC{Float64,Int}, H)
try
    M = Ma57(H,print_level=-1)
    println("*******   No problem in MA57 with CUTE")
catch
    println("*******   Problem in MA57 with CUTE")
end

try
    M2 = Ma57(H2,print_level=-1)
    println("*******   No problem in MA57 with sparse matrix")
catch
    println("*******   Problem in MA57 with  sparse matrix")
end
