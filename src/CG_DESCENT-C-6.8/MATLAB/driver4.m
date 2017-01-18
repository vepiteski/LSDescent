function driver4

    n = 100 ;
    x = ones(1, n) ; %initial guess

    Parm = cg_default ; %initialize default parameter values

    % make any changes to the default parameter values
    Parm.step = 1e-5 ;
    Parm.QuadStep = false ;
    Parm.rho = 1.5 ;

    % solve the problem
    x = cg_descent (x, 1.e-8, @myvalue, @mygrad, @myvalgrad, Parm) ;

    x = ones(1, n) ; %starting guess
    Parm.rho = 5 ;
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

    %first is function value,  second is the gradient vector
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
