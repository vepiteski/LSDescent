using Pkg
Pkg.activate(".")

using LinearAlgebra
using LinearOperators

include("Type.jl")

scaling = false

include("FormuleNOp.jl")  # rename push! to push2! for this unit test

include("FormuleN2Op.jl") # define push! (more efficient)

n=5
y = [1; 2.0; 3; 1; 1]
s = [-1; 2.0; 1; 1; 1]

B1 = InverseBFGSOperator(Float64, n; scaling = scaling)
B2 = InverseBFGSOperator(Float64, n; scaling = scaling)

B1 = push!(B1,y,s)
B2 = push2!(B2,y,s)

include("TypeChol.jl")
include("FormuleChOp.jl")

BCh1 = ChBFGSOperator(Float64, n; scaling = scaling)
BCh1 = push3!(BCh1,y,s)

using Test
@info "Premiers tests"
@test Matrix(B1) ≈ Matrix(B2)
@test Matrix(BCh1) ≈ Matrix(B1)


y = [-3;2.0;3;1;1]
s = [-1;2.0;-1;1;1]

B1 = push!(B1,y,s)
B2 = push2!(B2,y,s)
BCh1 = push3!(BCh1,y,s)

@info "Seconds tests"
@test Matrix(B1) ≈ Matrix(B2)
@test Matrix(BCh1) ≈ Matrix(B1)

let s=s, y=y, B1=B1, B2=B2,  BCh1 = BCh1
    for i=1:5
        @info i
        s = [s[5];s[1:4]]
        y = [y[5];y[1:4]]
        
        B1 = push!(B1,y,s)
        B2 = push2!(B2,y,s)
        BCh1 = push3!(BCh1,y,s)
        @info "B1=B2"
        @test Matrix(B1) ≈ Matrix(B2)
#halt
        @info "BCh1=B1"
        @test Matrix(BCh1) ≈ Matrix(B1)
    end
end
