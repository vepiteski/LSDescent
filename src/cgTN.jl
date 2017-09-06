using Compat
using LinearOperators


"Abstract type for statistics returned by a solver"
@compat abstract type KrylovStats end;

"Type for statistics returned by non-Lanczos solvers"
type SimpleStats <: KrylovStats
  solved :: Bool
  inconsistent :: Bool
  residuals :: Array{Float64,1}
  Aresiduals :: Array{Float64,1}
  status :: String
end

# A standard implementation of the Conjugate Gradient method.
# The only non-standard point about it is that it does not check
# that the operator is definite.
# It is possible to check that the system is inconsistent by
# monitoring ‖p‖, which would cost an extra norm computation per
# iteration.
#
# Dominique Orban, <dominique.orban@gerad.ca>
# Salt Lake City, UT, March 2015.
#
# Try to adapt a stopping criterion based on regulα for the ARCq.
#
# First reformulate the TR case using the characteristic function to confirm it is
# equivalent to the usual implementation. A must be symmetric to define a quadratic objective
# q(x) = 0.5*x'*A*x - b'*x
#
#   JPD february 09 2017, Montréal


export cgTN
# Methods for various argument types.
#include("cg_methods.jl")
cgTN{TA <: Number, Tb <: Number}(A :: Array{TA,2}, b :: Array{Tb,1};
                               atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6, itmax :: Int=0, verbose :: Bool=false) =
  cgTN(LinearOperator(A), b, atol=atol, rtol=rtol, itmax=itmax, verbose=verbose);

cgTN{TA <: Number, Tb <: Number, IA <: Integer}(A :: SparseMatrixCSC{TA,IA}, b :: Array{Tb,1};
                                              atol :: Float64=1.0e-8, rtol ::
                                              Float64=1.0e-6, itmax :: Int=0,  verbose :: Bool=false) =
  cgTN(LinearOperator(A), b, atol=atol, rtol=rtol, itmax=itmax, verbose=verbose);


"""The conjugate gradient method to solve the symmetric linear system Ax=b.

The method does _not_ abort if A is not definite.
"""
function cgTN{T <: Real}(A :: LinearOperator, b :: Array{T,1};
                         atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6, itmax :: Int=0,
                         verbose :: Bool=false)

    n = size(b, 1);
    (size(A, 1) == n & size(A, 2) == n) || error("Inconsistent problem size");
    #isequal(triu(A)',tril(A)) || error("Must supply Hermitian matrix")

    verbose && @printf("CG: system of %d equations in %d variables\n", n, n);

    # Initial state.
    x = zeros(n)
    x̂ = copy(x)

    γ = dot(b, b);
    γ == 0 && return x;
    r = copy(b);
    p = copy(r);

    σ = 0.0

    iter = 0;
    itmax == 0 && (itmax = 2 * n);

    rNorm = sqrt(γ);
    rNorms = [rNorm;];
    ε = atol + rtol * rNorm;
    verbose && @printf("%5d  %8.1e ", iter, rNorm)

    solved = rNorm <= ε;
    tired = iter >= itmax;
    neg_curv = false;
    status = "unknown";

    #q = s ->  0.5 * dot(s, copy(A * s)) - dot(b, s)

    #m = s ->  q(s) + norm(s)^3/(3*regulα)
    #hO = α -> m(x+α*p)
    Ap = copy(A * p);  # Bug in LinearOperators? A side effect spoils the computation without the copy.
    pAp = BLAS.dot(n, p, 1, Ap, 1);
    if pAp<=0
        neg_curv = true
        status = "gradient negative curvature"
        stats = SimpleStats(solved, false, rNorms, [], status);
        return (p, stats);
    else
        α = γ / pAp;
        BLAS.axpy!(n,  α,  p, 1, x, 1);  # Faster than x = x + σ * p;
        BLAS.axpy!(n, -α, Ap, 1, r, 1);  # Faster than r = r - α * Ap;
        γ_next = BLAS.dot(n, r, 1, r, 1);
        rNorm = sqrt(γ);
        push!(rNorms, rNorm);

        solved = (rNorm <= ε) ;
        tired = iter >= itmax;

        if !solved
            β = γ_next / γ;
            γ = γ_next;
            BLAS.scal!(n, β, p, 1)
            BLAS.axpy!(n, 1.0, r, 1, p, 1);  # Faster than p = r + β * p;
        end
        iter = iter + 1;
    end

    while ! (solved || tired || neg_curv)
        Ap = copy(A * p);  # Bug in LinearOperators? A side effect spoils the computation without the copy.
        pAp = BLAS.dot(n, p, 1, Ap, 1);

        α = γ / pAp;

        verbose && @printf("%8.1e  %7.1e  %7.1e\n", pAp, α, σ);

        # Move along p from x to the min if either
        # the next step leads farther than the min of the regularized model or
        # we have nonpositive curvature.
        if (pAp <= 0.0)
            α = 0.0
            neg_curv = true
            verbose &&println("negative curvature encountered ",iter," CG iterations.")
        else
            verbose && @printf("    %8.1e  %7.1e  %7.1e\n", pAp, α, σ);
            BLAS.axpy!(n,  α,  p, 1, x, 1);  # Faster than x = x + σ * p;
            BLAS.axpy!(n, -α, Ap, 1, r, 1);  # Faster than r = r - α * Ap;
            γ_next = BLAS.dot(n, r, 1, r, 1);
            rNorm = sqrt(γ);
            push!(rNorms, rNorm);

            solved = (rNorm <= ε) ;
            tired = iter >= itmax;
        end

        if !(solved || neg_curv)
            β = γ_next / γ;
            γ = γ_next;
            BLAS.scal!(n, β, p, 1)
            BLAS.axpy!(n, 1.0, r, 1, p, 1);  # Faster than p = r + β * p;
        end
        iter = iter + 1;
        verbose && @printf("%5d  %8.1e ", iter, rNorm);
    end
    verbose && @printf("\n");

  status = neg_curv ? "negative curvature avoided" : (tired ? "maximum number of iterations exceeded" : "solution good enough given atol and rtol")
  stats = SimpleStats(solved, false, rNorms, [], status);
  return (x, stats);
end
