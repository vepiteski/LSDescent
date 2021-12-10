Préparation pour intégrer diverses variantes de mises-à-jour quasi-Newton dans LSDescent. Pas le temps de les intégrer ce soir.

On encapsule la représentation matricielle dans un LinearOperator. Les types requis sont dans Type.jl.

UnitTests.jl   teste que les représentations "full matrix" et L-BFGS donnent la même chose

runtests.jl applique ces variantes à des problèmes de la collection OptimizationProblems.jl

FormuleN2 est la formule O(n^2) et FormuleN celle mathématique O(n^3) et les versions Op sont implémentées sur les operateurs.


Un mot d'attention, ATTENTION, si B est un opérateur tel que défini dans Type.jl, l'expression
       - B*g   N'APPLIQUE PAS le - UNAIRE

IL FAUT ÉCRIRE    -(B*g)


JPD 7 décembre 2021

