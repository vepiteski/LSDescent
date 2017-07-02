export C1LineFunction2, C2LineFunction2, AbstractLineFunction2
export obj, grad, grad!, hess

# Import methods that we extend.
import NLPModels.obj, NLPModels.grad, NLPModels.grad!, NLPModels.hess


"""A type to represent the restriction of a C1 function to a direction.
Given f : R → Rⁿ, x ∈ Rⁿ and a nonzero direction d ∈ Rⁿ,

    ϕ = C1LineFunction(f, x, d)

represents the function ϕ : R → R defined by

    ϕ(t) := f(x + td).
"""
type C1LineFunction2
  nlp :: AbstractNLPModel
  x :: Vector
  d :: Vector
  f_eval :: Int64
  g_eval :: Int64
  h_eval :: Int64

    function C1LineFunction2(nlp :: AbstractNLPModel, x :: Vector, d :: Vector;
                             f_eval ::Int64 = 0, g_eval :: Int64 = 0, h_eval :: Int64 = 0)
      return new(nlp, x, d, f_eval, g_eval, h_eval)
    end

end


"""`obj(f, t)` evaluates the objective of the `C1LineFunction`

    ϕ(t) := f(x + td).
"""
function obj(f :: C1LineFunction2, t :: Float64)
  f.f_eval += 1
  return obj(f.nlp, f.x + t * f.d)
end


"""`grad(f, t)` evaluates the first derivative of the `C1LineFunction`

    ϕ(t) := f(x + td),

i.e.,

    ϕ'(t) = ∇f(x + td)ᵀd.
"""
function grad(f :: C1LineFunction2, t :: Float64)
  f.g_eval += 1
  return dot(grad(f.nlp, f.x + t * f.d), f.d)
end

"""`grad!(f, t, g)` evaluates the first derivative of the `C1LineFunction`

    ϕ(t) := f(x + td),

i.e.,

    ϕ'(t) = ∇f(x + td)ᵀd.

The gradient ∇f(x + td) is stored in `g`.
"""
function grad!(f :: C1LineFunction2, t :: Float64, g :: Vector{Float64})
  f.g_eval += 1
  return dot(grad!(f.nlp, f.x + t * f.d, g), f.d)
end

"""A type to represent the restriction of a C2 function to a direction.
Given f : R → Rⁿ, x ∈ Rⁿ and a nonzero direction d ∈ Rⁿ,

    ϕ = C2LineFunction(f, x, d)

represents the function ϕ : R → R defined by

    ϕ(t) := f(x + td).
"""
type C2LineFunction2
  nlp :: AbstractNLPModel
  x :: Vector
  d :: Vector
  f_eval :: Int64
  g_eval :: Int64
  h_eval :: Int64

    function C2LineFunction2(nlp :: AbstractNLPModel, x :: Vector, d :: Vector;
                             f_eval ::Int64 = 0, g_eval :: Int64 = 0, h_eval :: Int64 = 0)
      return new(nlp, x, d, f_eval, g_eval, h_eval)
    end
end


"""`obj(f, t)` evaluates the objective of the `C2LineFunction`

    ϕ(t) := f(x + td).
"""
function obj(f :: C2LineFunction2, t :: Float64)
  f.f_eval += 1
  return obj(f.nlp, f.x + t * f.d)
end

"""`grad(f, t)` evaluates the first derivative of the `C2LineFunction`

    ϕ(t) := f(x + td),

i.e.,

    ϕ'(t) = ∇f(x + td)ᵀd.
"""
function grad(f :: C2LineFunction2, t :: Float64)
  f.g_eval += 1
  return dot(grad(f.nlp, f.x + t * f.d), f.d)
end

"""`grad!(f, t, g)` evaluates the first derivative of the `C2LineFunction`

    ϕ(t) := f(x + td),

i.e.,

    ϕ'(t) = ∇f(x + td)ᵀd.

The gradient ∇f(x + td) is stored in `g`.
"""
function grad!(f :: C2LineFunction2, t :: Float64, g :: Vector{Float64})
  f.g_eval += 1
  return dot(grad!(f.nlp, f.x + t * f.d, g), f.d)
end


"""Evaluate the second derivative of the `C2LineFunction`

    ϕ(t) := f(x + td),

i.e.,

    ϕ"(t) = dᵀ∇²f(x + td)d.
"""
function hess(f :: C2LineFunction2, t :: Float64)
  f.h_eval += 1
  return dot(f.d, hprod(f.nlp, f.x + t * f.d, f.d))
end

AbstractLineFunction2 = Union{C1LineFunction2, C2LineFunction2}
