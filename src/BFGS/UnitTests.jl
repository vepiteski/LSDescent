using Pkg
Pkg.activate(".")

using LinearAlgebra
using LinearOperators

include("Type.jl")
include("FormuleN2Op.jl") # define push! (more efficient)

include("TypeCompact.jl")

n=5
y = [1; 2.0; 3; 1; 1]
s = [-1; 2.0; 1; 1; 1]

B1 = InverseBFGSOperator(Float64, n; scaling = false)
B1 = push!(B1,y,s)

LB = InverseLBFGSOperator(5, scaling = false, mem = 7);
LB = push!(LB,y,s);

CB = CompactInverseBFGSOperator(5, scaling = false, mem = 7);
CB = push!(CB,y,s);

using Test
@info "Premiers tests"
@test Matrix(B1) ≈ Matrix(LB)
@test Matrix(B1) ≈ Matrix(CB)

y = [-3;2.0;3;1;1]
s = [-1;2.0;-1;1;1]

B1 = push!(B1,y,s);
LB = push!(LB,y,s);
CB = push!(CB,y,s);

@info "Seconds tests"
@test Matrix(B1) ≈ Matrix(LB)
@test Matrix(B1) ≈ Matrix(CB)

let s=s, y=y, B1=B1, LB = LB, CB = CB
    for i=1:5
        @info i
        s = [s[5];s[1:4]]
        y = [y[5];y[1:4]]

        B1 = push!(B1,y,s);
        LB = push!(LB,y,s);
        CB = push!(CB,y,s);

        @info "B1=LB"
        @test Matrix(B1) ≈ Matrix(LB)
        @info "B1=CB"
        @test Matrix(B1) ≈ Matrix(CB)
    end
end

@info "Third tests - Testing multiply! Operator"
x = rand(5)

d1 = B1 * x
d2 = LB * x
d3 = CB * x


@info "dB=dLB"
@test d1 ≈ d2
@info "dB=dCB"
@test d1 ≈ d3

d1 = - B1 * x
d2 = - LB * x
d3 = - CB * x


@info "dB=dLB"
@test d1 ≈ d2
@info "dB=dCB"
@test d1 ≈ d3

p1 = - (B1 * x)
p2 = - (LB * x)
p3 = - (CB * x)


@info "pB=pLB"
@test p1 ≈ p2
@info "pB=pCB"
@test p1 ≈ p3


@info "dB=pB"
@test d1 ≈ p1
@info "dLB=pLB"
@test d2 ≈ p2
@info "dCB=pCB"
@test d3 ≈ p3