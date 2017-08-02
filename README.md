# LSDescentMethods

Several line search descent methods for unconstrained optimization.

## Installing
`julia> Pkg.clone("https://github.com/vepiteski/LSDescentMethods.git")`

## Stopping
The stopping conditions for the algorithms are controlled by the package
[Stopping](https://github.com/vepiteski/Stopping.jl). Versions of the algorithms
not using Stopping are in the "Old" folder.

## Line Search
An Armijo backtracking process is provided directly within the package. It is the
default line search algorithm used by the descent algorithms. A larger
collection of line search algorithms is provided by the package
[LineSearch](https://github.com/Goysa2/LineSearch).  
