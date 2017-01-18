function driver2
 
    n = 100 ;
    x = ones(n, 1) ; %initial guess
 
    Parm = cg_default ; %initialize default parameter values
    
    % do not perform the quadratic interpolation step
    Parm.QuadStep = false ;
 
    % solve the problem
    x = cg_descent (x, 1.e-8, @myvalue, @mygrad, @myvalgrad, Parm) ;
    
    % now perform the quadratic interpolation step
    x = ones(1, n) ; %starting guess
    Parm.QuadStep = true ;
    %solve the problem
    x = cg_descent (x, 1.e-8, @myvalue, @mygrad, @myvalgrad, Parm) ;

    function f = myvalue(x)
       f = 0 ;
       n = length(x) ;
       for i=1:n
           t = i^0.5 ;
           f = f + exp(x(i)) - t*x(i) ;
       end
    end

    function g = mygrad(x)
        n = length(x) ;
        g = zeros (1, n) ;
        for i=1:n
           t = i^0.5 ;
           g(i) = exp(x(i))-t ;
        end
    end

    %first is function value, second is the gradient vector
    function [f, g] = myvalgrad(x)
        n = length(x) ;
        f = 0 ;
        g = zeros (1, n) ;
        for i=1:n
           t = i^0.5 ;
           ex = exp(x(i)) ;
           f = f + ex - t*x(i) ;
           g(i) = ex - t ;
        end
    end

end
