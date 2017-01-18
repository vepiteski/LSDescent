function driver1
 
    n = 100;

    x = ones(1, n) ; % initial guess
    % call cg_descent with default parameter values
    
    x = cg_descent (x, 1.e-8, @myvalue, @mygrad, @myvalgrad);
    
    % to change parameter values, first load default values
    Parm = cg_default () ;
    % change any of the default parameter values
    Parm.PrintLevel = 2;

    x = ones(1,n) ; % initial guess
    % solve the problem using valgrad and PrintLevel 2
    x = cg_descent (x, 1.e-8, @myvalue, @mygrad, @myvalgrad, Parm) ;
    
    x = ones(1,n) ; % initial guess
    % solve the problem using PrintLevel 2, but without valgrad
    x = cg_descent (x, 1.e-8, @myvalue, @mygrad, Parm) ;
    
    function f = myvalue(x)
       f = 0;
       n = length(x) ;
       for i=1:n
           t = i^0.5 ;
           f = f + exp(x(i)) - t*x(i) ;
       end
    end
    
    function g = mygrad(x)
        n = length(x) ;
        g = zeros (1,n) ;
        for i=1:n
           t = i^0.5 ;
           g(i) = exp(x(i))-t ;
        end
    end
    
    %first is function value, second is the gradient vector
    function [f,g] = myvalgrad(x)
        n = length(x) ;
        f = 0 ;
        g = zeros (1,n) ;
        for i=1:n
           t = i^0.5 ;
           ex = exp(x(i)) ;
           f = f + ex - t*x(i) ;
           g(i) = ex - t ;
        end
    end

end
