# LSDescent

Several line search descent methods for unconstrained optimization. Currently, variants of the conjugate gradient, quasi-Newton (BFGS) either using full matrices or the Limited memory version and Newton's method using the spectral factorization are provided.

## Installing
`julia> Pkg.add("https://github.com/vepiteski/LSDescent.git")`


## Stopping
The stopping conditions for the algorithms are controlled by the package
[Stopping](https://github.com/vepiteski/Stopping.jl). 

## Line Search
Line searches are defined in https://github.com/vepiteski/OneDmin.git


## Known issues
- The bfgs and L-bfgs will need to be updated to uniformize the possibility of specifying the "scale" parameter.
- The wrappers for conjugate gradient variants are currently not working, needing updating.