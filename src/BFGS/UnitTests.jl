using Pkg
Pkg.activate(".")

using LinearAlgebra

include("FormuleNOp.jl")  # rename push! to push2! for this unit test
push2! = push!  # ça ne marche pas de renommer... 
include("FormuleN2Op.jl") # define push! (more efficient)

using LinearOperators
n=5
y = [1; 2.0; 3; 1; 1]
s = [-1; 2.0; 1; 1; 1]

B1 = InverseBFGSOperator(Float64, n; scaling = false)
B2 = InverseBFGSOperator(Float64, n; scaling = false)

B1 = push!(B1,y,s)
B2 = push2!(B2,y,s)



LB = InverseLBFGSOperator(5, scaling = false, mem = 7);
LB = push!(LB,y,s);

using Test
@info "Premiers tests"
@test Matrix(B1) ≈ Matrix(B2)
@test Matrix(B1) ≈ Matrix(LB)


y = [-3;2.0;3;1;1]
s = [-1;2.0;-1;1;1]

B1 = push!(B1,y,s)
B2 = push2!(B2,y,s)
LB = push!(LB,y,s);

@info "Seconds tests"
@test Matrix(B1) ≈ Matrix(B2)
@test Matrix(B1) ≈ Matrix(LB)

let s=s, y=y, B1=B1, B2=B2, LB = LB
    for i=1:5
        @info i
        s = [s[5];s[1:4]]
        y = [y[5];y[1:4]]
        
        B1 = push!(B1,y,s)
        B2 = push2!(B2,y,s)
        LB = push!(LB,y,s);
        
        @info "B1=B2"
        @test Matrix(B1) ≈ Matrix(B2)
        @info "B1=LB"
        @test Matrix(B1) ≈ Matrix(LB)
    end
end
